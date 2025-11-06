---
# Auto-generated front matter
Title: Observer
LastUpdated: 2025-11-06T20:45:58.522099
Tags: []
Status: draft
---

# Observer Pattern

## Pattern Name & Intent

**Observer** - Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

The Observer pattern establishes a dependency relationship between a subject (observable) and multiple observers. When the subject's state changes, it automatically notifies all registered observers, allowing them to react to the change. This pattern is fundamental to event-driven architectures and is widely used in modern applications.

## When to Use

### Appropriate Scenarios

- **Event-Driven Systems**: When you need to decouple event producers from consumers
- **Model-View Architecture**: When views need to update when model changes
- **Notification Systems**: When multiple components need to be notified of changes
- **Real-time Updates**: When you need to push updates to multiple clients
- **Plugin Systems**: When you need to allow dynamic registration of handlers

### When NOT to Use

- **Simple Callbacks**: When you only need a single callback
- **Performance Critical**: When notification overhead is too high
- **Tight Coupling**: When observers need to know about each other
- **Memory Leaks**: When observers are not properly unregistered

## Real-World Use Cases (Fintech/Payments)

### Payment Event System

```go
// Event types
type PaymentEventType string

const (
    PaymentCreated   PaymentEventType = "payment_created"
    PaymentProcessed PaymentEventType = "payment_processed"
    PaymentFailed    PaymentEventType = "payment_failed"
    PaymentRefunded  PaymentEventType = "payment_refunded"
)

// Event data
type PaymentEvent struct {
    Type        PaymentEventType `json:"type"`
    PaymentID   string           `json:"payment_id"`
    Amount      float64          `json:"amount"`
    Currency    string           `json:"currency"`
    UserID      string           `json:"user_id"`
    MerchantID  string           `json:"merchant_id"`
    Timestamp   time.Time        `json:"timestamp"`
    Metadata    map[string]interface{} `json:"metadata"`
}

// Observer interface
type PaymentObserver interface {
    OnPaymentEvent(event *PaymentEvent)
    GetObserverID() string
}

// Subject interface
type PaymentEventSubject interface {
    RegisterObserver(observer PaymentObserver)
    UnregisterObserver(observerID string)
    NotifyObservers(event *PaymentEvent)
}

// Concrete subject
type PaymentEventManager struct {
    observers map[string]PaymentObserver
    mutex     sync.RWMutex
}

func NewPaymentEventManager() *PaymentEventManager {
    return &PaymentEventManager{
        observers: make(map[string]PaymentObserver),
    }
}

func (pem *PaymentEventManager) RegisterObserver(observer PaymentObserver) {
    pem.mutex.Lock()
    defer pem.mutex.Unlock()
    pem.observers[observer.GetObserverID()] = observer
}

func (pem *PaymentEventManager) UnregisterObserver(observerID string) {
    pem.mutex.Lock()
    defer pem.mutex.Unlock()
    delete(pem.observers, observerID)
}

func (pem *PaymentEventManager) NotifyObservers(event *PaymentEvent) {
    pem.mutex.RLock()
    defer pem.mutex.RUnlock()

    for _, observer := range pem.observers {
        go observer.OnPaymentEvent(event) // Async notification
    }
}

// Concrete observers
type AuditLogger struct {
    logger *log.Logger
}

func NewAuditLogger() *AuditLogger {
    file, _ := os.OpenFile("audit.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
    return &AuditLogger{
        logger: log.New(file, "AUDIT: ", log.LstdFlags),
    }
}

func (al *AuditLogger) OnPaymentEvent(event *PaymentEvent) {
    al.logger.Printf("Payment Event: %s - ID: %s, Amount: %.2f %s",
        event.Type, event.PaymentID, event.Amount, event.Currency)
}

func (al *AuditLogger) GetObserverID() string {
    return "audit_logger"
}

type NotificationService struct {
    emailService EmailService
    smsService   SMSService
}

func (ns *NotificationService) OnPaymentEvent(event *PaymentEvent) {
    switch event.Type {
    case PaymentProcessed:
        ns.emailService.SendPaymentConfirmation(event.UserID, event.PaymentID)
    case PaymentFailed:
        ns.smsService.SendPaymentFailureAlert(event.UserID, event.PaymentID)
    }
}

func (ns *NotificationService) GetObserverID() string {
    return "notification_service"
}
```

### Order Status Updates

```go
type OrderStatus string

const (
    OrderPending    OrderStatus = "pending"
    OrderConfirmed  OrderStatus = "confirmed"
    OrderShipped    OrderStatus = "shipped"
    OrderDelivered  OrderStatus = "delivered"
    OrderCancelled  OrderStatus = "cancelled"
)

type OrderEvent struct {
    OrderID    string      `json:"order_id"`
    Status     OrderStatus `json:"status"`
    UserID     string      `json:"user_id"`
    Timestamp  time.Time   `json:"timestamp"`
    Metadata   map[string]interface{} `json:"metadata"`
}

type OrderObserver interface {
    OnOrderStatusChange(event *OrderEvent)
    GetObserverID() string
}

type OrderEventManager struct {
    observers map[string]OrderObserver
    mutex     sync.RWMutex
}

func (oem *OrderEventManager) NotifyOrderStatusChange(orderID string, status OrderStatus, userID string) {
    event := &OrderEvent{
        OrderID:   orderID,
        Status:    status,
        UserID:    userID,
        Timestamp: time.Now(),
    }

    oem.mutex.RLock()
    defer oem.mutex.RUnlock()

    for _, observer := range oem.observers {
        go observer.OnOrderStatusChange(event)
    }
}
```

## Go Implementation

### Generic Observer Pattern

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Generic event interface
type Event interface {
    GetType() string
    GetTimestamp() time.Time
}

// Generic observer interface
type Observer[T Event] interface {
    OnEvent(event T)
    GetID() string
}

// Generic subject interface
type Subject[T Event] interface {
    RegisterObserver(observer Observer[T])
    UnregisterObserver(observerID string)
    NotifyObservers(event T)
}

// Generic event manager
type EventManager[T Event] struct {
    observers map[string]Observer[T]
    mutex     sync.RWMutex
    ctx       context.Context
    cancel    context.CancelFunc
}

func NewEventManager[T Event]() *EventManager[T] {
    ctx, cancel := context.WithCancel(context.Background())
    return &EventManager[T]{
        observers: make(map[string]Observer[T]),
        ctx:       ctx,
        cancel:    cancel,
    }
}

func (em *EventManager[T]) RegisterObserver(observer Observer[T]) {
    em.mutex.Lock()
    defer em.mutex.Unlock()
    em.observers[observer.GetID()] = observer
}

func (em *EventManager[T]) UnregisterObserver(observerID string) {
    em.mutex.Lock()
    defer em.mutex.Unlock()
    delete(em.observers, observerID)
}

func (em *EventManager[T]) NotifyObservers(event T) {
    em.mutex.RLock()
    defer em.mutex.RUnlock()

    for _, observer := range em.observers {
        go func(obs Observer[T]) {
            select {
            case <-em.ctx.Done():
                return
            default:
                obs.OnEvent(event)
            }
        }(observer)
    }
}

func (em *EventManager[T]) Close() {
    em.cancel()
}

// Concrete event types
type UserEvent struct {
    Type      string    `json:"type"`
    UserID    string    `json:"user_id"`
    Timestamp time.Time `json:"timestamp"`
    Data      map[string]interface{} `json:"data"`
}

func (ue *UserEvent) GetType() string {
    return ue.Type
}

func (ue *UserEvent) GetTimestamp() time.Time {
    return ue.Timestamp
}

// Concrete observers
type UserActivityLogger struct {
    logger *log.Logger
}

func NewUserActivityLogger() *UserActivityLogger {
    return &UserActivityLogger{
        logger: log.New(os.Stdout, "USER_ACTIVITY: ", log.LstdFlags),
    }
}

func (ual *UserActivityLogger) OnEvent(event *UserEvent) {
    ual.logger.Printf("User %s: %s", event.UserID, event.Type)
}

func (ual *UserActivityLogger) GetID() string {
    return "user_activity_logger"
}

type UserAnalyticsTracker struct {
    analyticsService AnalyticsService
}

func (uat *UserAnalyticsTracker) OnEvent(event *UserEvent) {
    uat.analyticsService.TrackUserEvent(event.UserID, event.Type, event.Data)
}

func (uat *UserAnalyticsTracker) GetID() string {
    return "user_analytics_tracker"
}
```

### Event Bus Implementation

```go
type EventBus struct {
    subscribers map[string][]Observer[Event]
    mutex       sync.RWMutex
    channels    map[string]chan Event
}

func NewEventBus() *EventBus {
    return &EventBus{
        subscribers: make(map[string][]Observer[Event]),
        channels:    make(map[string]chan Event),
    }
}

func (eb *EventBus) Subscribe(eventType string, observer Observer[Event]) {
    eb.mutex.Lock()
    defer eb.mutex.Unlock()

    eb.subscribers[eventType] = append(eb.subscribers[eventType], observer)

    // Create channel if it doesn't exist
    if _, exists := eb.channels[eventType]; !exists {
        eb.channels[eventType] = make(chan Event, 100)
        go eb.processEvents(eventType)
    }
}

func (eb *EventBus) Unsubscribe(eventType string, observerID string) {
    eb.mutex.Lock()
    defer eb.mutex.Unlock()

    if observers, exists := eb.subscribers[eventType]; exists {
        for i, obs := range observers {
            if obs.GetID() == observerID {
                eb.subscribers[eventType] = append(observers[:i], observers[i+1:]...)
                break
            }
        }
    }
}

func (eb *EventBus) Publish(event Event) {
    eb.mutex.RLock()
    channel, exists := eb.channels[event.GetType()]
    eb.mutex.RUnlock()

    if exists {
        select {
        case channel <- event:
        default:
            // Channel is full, drop event or handle overflow
            log.Printf("Event channel full, dropping event: %s", event.GetType())
        }
    }
}

func (eb *EventBus) processEvents(eventType string) {
    channel := eb.channels[eventType]

    for event := range channel {
        eb.mutex.RLock()
        observers := eb.subscribers[eventType]
        eb.mutex.RUnlock()

        for _, observer := range observers {
            go observer.OnEvent(event)
        }
    }
}
```

## Variants & Trade-offs

### Variants

#### 1. Push Model

```go
type PushObserver interface {
    OnEvent(eventType string, data interface{})
}
```

**Pros**: Simple, direct notification
**Cons**: Observers must handle all event types

#### 2. Pull Model

```go
type PullObserver interface {
    Update(subject Subject)
}

func (obs *PullObserver) Update(subject Subject) {
    // Observer pulls data from subject
    data := subject.GetData()
    obs.ProcessData(data)
}
```

**Pros**: Observers control what data they need
**Cons**: More complex, potential performance issues

#### 3. Event Sourcing

```go
type EventStore interface {
    AppendEvent(event Event) error
    GetEvents(aggregateID string) ([]Event, error)
}

type EventSourcedObserver struct {
    eventStore EventStore
}

func (eso *EventSourcedObserver) OnEvent(event Event) {
    eso.eventStore.AppendEvent(event)
}
```

**Pros**: Complete audit trail, replay capability
**Cons**: Storage overhead, complexity

### Trade-offs

| Aspect              | Pros                              | Cons                                   |
| ------------------- | --------------------------------- | -------------------------------------- |
| **Decoupling**      | Loose coupling between components | Can become hard to track dependencies  |
| **Flexibility**     | Easy to add/remove observers      | Can lead to performance issues         |
| **Testing**         | Easy to mock observers            | Complex event flows hard to test       |
| **Performance**     | Async processing possible         | Memory overhead for observers          |
| **Maintainability** | Clear separation of concerns      | Can become complex with many observers |

## Testable Example

```go
package main

import (
    "testing"
    "time"
)

// Mock observer for testing
type MockObserver struct {
    ID       string
    events   []Event
    mutex    sync.Mutex
}

func NewMockObserver(id string) *MockObserver {
    return &MockObserver{
        ID:     id,
        events: make([]Event, 0),
    }
}

func (mo *MockObserver) OnEvent(event Event) {
    mo.mutex.Lock()
    defer mo.mutex.Unlock()
    mo.events = append(mo.events, event)
}

func (mo *MockObserver) GetID() string {
    return mo.ID
}

func (mo *MockObserver) GetEvents() []Event {
    mo.mutex.Lock()
    defer mo.mutex.Unlock()
    return append([]Event(nil), mo.events...)
}

func (mo *MockObserver) ClearEvents() {
    mo.mutex.Lock()
    defer mo.mutex.Unlock()
    mo.events = make([]Event, 0)
}

// Tests
func TestEventManager_RegisterObserver(t *testing.T) {
    em := NewEventManager[*UserEvent]()
    observer := NewMockObserver("test_observer")

    em.RegisterObserver(observer)

    // Verify observer is registered
    em.mutex.RLock()
    _, exists := em.observers["test_observer"]
    em.mutex.RUnlock()

    if !exists {
        t.Error("Observer should be registered")
    }
}

func TestEventManager_NotifyObservers(t *testing.T) {
    em := NewEventManager[*UserEvent]()
    observer := NewMockObserver("test_observer")

    em.RegisterObserver(observer)

    event := &UserEvent{
        Type:      "user_login",
        UserID:    "user_123",
        Timestamp: time.Now(),
        Data:      map[string]interface{}{"ip": "192.168.1.1"},
    }

    em.NotifyObservers(event)

    // Wait for async processing
    time.Sleep(100 * time.Millisecond)

    events := observer.GetEvents()
    if len(events) != 1 {
        t.Errorf("Expected 1 event, got %d", len(events))
    }

    if events[0].GetType() != "user_login" {
        t.Errorf("Expected event type 'user_login', got %s", events[0].GetType())
    }
}

func TestEventManager_UnregisterObserver(t *testing.T) {
    em := NewEventManager[*UserEvent]()
    observer := NewMockObserver("test_observer")

    em.RegisterObserver(observer)
    em.UnregisterObserver("test_observer")

    event := &UserEvent{
        Type:      "user_login",
        UserID:    "user_123",
        Timestamp: time.Now(),
    }

    em.NotifyObservers(event)

    // Wait for async processing
    time.Sleep(100 * time.Millisecond)

    events := observer.GetEvents()
    if len(events) != 0 {
        t.Errorf("Expected 0 events after unregister, got %d", len(events))
    }
}

func TestEventBus_PublishSubscribe(t *testing.T) {
    bus := NewEventBus()
    observer := NewMockObserver("test_observer")

    bus.Subscribe("user_event", observer)

    event := &UserEvent{
        Type:      "user_event",
        UserID:    "user_123",
        Timestamp: time.Now(),
    }

    bus.Publish(event)

    // Wait for async processing
    time.Sleep(100 * time.Millisecond)

    events := observer.GetEvents()
    if len(events) != 1 {
        t.Errorf("Expected 1 event, got %d", len(events))
    }
}

func TestMultipleObservers(t *testing.T) {
    em := NewEventManager[*UserEvent]()

    observer1 := NewMockObserver("observer_1")
    observer2 := NewMockObserver("observer_2")

    em.RegisterObserver(observer1)
    em.RegisterObserver(observer2)

    event := &UserEvent{
        Type:      "user_action",
        UserID:    "user_123",
        Timestamp: time.Now(),
    }

    em.NotifyObservers(event)

    // Wait for async processing
    time.Sleep(100 * time.Millisecond)

    events1 := observer1.GetEvents()
    events2 := observer2.GetEvents()

    if len(events1) != 1 || len(events2) != 1 {
        t.Errorf("Expected both observers to receive 1 event, got %d and %d",
            len(events1), len(events2))
    }
}
```

## Integration Tips

### 1. With Context and Cancellation

```go
type ContextualObserver struct {
    ctx    context.Context
    cancel context.CancelFunc
}

func (co *ContextualObserver) OnEvent(event Event) {
    select {
    case <-co.ctx.Done():
        return
    default:
        co.processEvent(event)
    }
}
```

### 2. With Error Handling

```go
type ErrorHandlingObserver struct {
    observer Observer
    logger   *log.Logger
}

func (eho *ErrorHandlingObserver) OnEvent(event Event) {
    defer func() {
        if r := recover(); r != nil {
            eho.logger.Printf("Observer panic: %v", r)
        }
    }()

    if err := eho.observer.OnEvent(event); err != nil {
        eho.logger.Printf("Observer error: %v", err)
    }
}
```

### 3. With Metrics and Monitoring

```go
type MetricsObserver struct {
    observer Observer
    metrics  MetricsCollector
}

func (mo *MetricsObserver) OnEvent(event Event) {
    start := time.Now()
    defer func() {
        mo.metrics.RecordObserverLatency(event.GetType(), time.Since(start))
    }()

    mo.observer.OnEvent(event)
    mo.metrics.IncrementObserverEvents(event.GetType())
}
```

## Common Interview Questions

### 1. What is the Observer pattern and how does it work?

**Answer**: The Observer pattern defines a one-to-many dependency between objects. When the subject's state changes, it automatically notifies all registered observers. It's implemented using interfaces for observers and a registration/notification mechanism in the subject.

### 2. How do you implement the Observer pattern in Go?

**Answer**: Define observer interfaces, implement concrete observers, create a subject that maintains a list of observers, and provide methods to register/unregister observers and notify them of changes.

### 3. What are the benefits and drawbacks of the Observer pattern?

**Answer**: Benefits include loose coupling, dynamic relationships, and support for broadcast communication. Drawbacks include potential memory leaks, unexpected updates, and performance issues with many observers.

### 4. How do you handle errors in the Observer pattern?

**Answer**: Use error handling wrappers, implement circuit breakers for failing observers, use async processing with proper error handling, and implement retry mechanisms for critical observers.

### 5. How do you prevent memory leaks with the Observer pattern?

**Answer**: Always unregister observers when they're no longer needed, use weak references where possible, implement proper cleanup in observer destructors, and use context cancellation for long-running observers.
