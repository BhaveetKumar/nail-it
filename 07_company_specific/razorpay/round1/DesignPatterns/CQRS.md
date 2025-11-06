---
# Auto-generated front matter
Title: Cqrs
LastUpdated: 2025-11-06T20:45:58.527385
Tags: []
Status: draft
---

# CQRS (Command Query Responsibility Segregation) Pattern

## Pattern Name & Intent

**CQRS (Command Query Responsibility Segregation)** is an architectural pattern that separates read and write operations for data storage. It uses separate models to update information (commands) and read information (queries).

**Key Intent:**

- Separate read and write concerns for better scalability
- Optimize read and write operations independently
- Enable different data models for commands and queries
- Support eventual consistency between read and write models
- Allow independent scaling of read and write workloads

## When to Use

**Use CQRS when:**

1. **Complex Domain Logic**: Domain has complex business rules that differ between reads and writes
2. **Different Read/Write Patterns**: Read and write workloads have significantly different characteristics
3. **Performance Requirements**: Need to optimize read and write operations separately
4. **Scalability Needs**: Read and write operations need independent scaling
5. **Event-Driven Architecture**: Already using event sourcing or event-driven patterns
6. **Reporting Requirements**: Complex reporting that differs from transactional data structure

**Don't use when:**

- Simple CRUD operations are sufficient
- Read and write models are nearly identical
- Team lacks experience with distributed systems
- Eventual consistency is not acceptable
- System complexity isn't justified by benefits

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Processing System

```go
// Write model - optimized for transactions
type PaymentCommand struct {
    PaymentID   string
    Amount      decimal.Decimal
    Currency    string
    FromAccount string
    ToAccount   string
    Reference   string
}

// Read model - optimized for queries and reporting
type PaymentView struct {
    PaymentID     string
    Amount        decimal.Decimal
    Currency      string
    Status        string
    ProcessedAt   time.Time
    FromAccount   AccountSummary
    ToAccount     AccountSummary
    FeeAmount     decimal.Decimal
    Description   string
    Tags          []string
}
```

### 2. Trading Platform

```go
// Command model - focus on trade execution
type PlaceOrderCommand struct {
    OrderID    string
    UserID     string
    Symbol     string
    Quantity   int64
    Price      decimal.Decimal
    OrderType  OrderType
    TimeInForce TimeInForce
}

// Query model - optimized for market data and portfolio views
type OrderBookView struct {
    Symbol    string
    BidOrders []PriceLevel
    AskOrders []PriceLevel
    LastTrade TradeInfo
    Stats     MarketStats
}

type PortfolioView struct {
    UserID       string
    Positions    []Position
    TotalValue   decimal.Decimal
    DayPnL       decimal.Decimal
    OpenOrders   []OrderSummary
    TradingPower decimal.Decimal
}
```

### 3. Banking Transaction System

```go
// Command model - focused on account operations
type TransferCommand struct {
    TransactionID string
    FromAccountID string
    ToAccountID   string
    Amount        decimal.Decimal
    Purpose       string
    Metadata      map[string]string
}

// Query model - optimized for account statements and analytics
type TransactionView struct {
    TransactionID   string
    Date           time.Time
    Description    string
    Amount         decimal.Decimal
    RunningBalance decimal.Decimal
    Category       string
    Tags           []string
    Location       *GeoLocation
}

type AccountSummaryView struct {
    AccountID      string
    Balance        decimal.Decimal
    AvailableBalance decimal.Decimal
    RecentTransactions []TransactionView
    MonthlySpending    SpendingBreakdown
}
```

### 4. Fraud Detection System

```go
// Command model - simple fraud case creation
type CreateFraudCaseCommand struct {
    CaseID        string
    TransactionID string
    UserID        string
    RiskScore     float64
    Reason        string
}

// Query model - rich fraud analytics
type FraudAnalyticsView struct {
    CaseID          string
    RiskScore       float64
    UserProfile     UserRiskProfile
    TransactionHistory []SuspiciousTransaction
    SimilarCases    []RelatedCase
    MLPredictions   []MLScore
    InvestigationNotes []Note
}
```

## Go Implementation

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "time"
    "sync"
    "github.com/shopspring/decimal"
    "github.com/google/uuid"
)

// Domain Events
type Event interface {
    GetEventID() string
    GetAggregateID() string
    GetEventType() string
    GetTimestamp() time.Time
    GetVersion() int
}

type BaseEvent struct {
    EventID     string    `json:"event_id"`
    AggregateID string    `json:"aggregate_id"`
    EventType   string    `json:"event_type"`
    Timestamp   time.Time `json:"timestamp"`
    Version     int       `json:"version"`
}

func (e *BaseEvent) GetEventID() string     { return e.EventID }
func (e *BaseEvent) GetAggregateID() string { return e.AggregateID }
func (e *BaseEvent) GetEventType() string   { return e.EventType }
func (e *BaseEvent) GetTimestamp() time.Time { return e.Timestamp }
func (e *BaseEvent) GetVersion() int        { return e.Version }

// Payment Domain Events
type PaymentInitiatedEvent struct {
    BaseEvent
    PaymentID   string          `json:"payment_id"`
    Amount      decimal.Decimal `json:"amount"`
    Currency    string          `json:"currency"`
    FromAccount string          `json:"from_account"`
    ToAccount   string          `json:"to_account"`
    Reference   string          `json:"reference"`
}

type PaymentProcessedEvent struct {
    BaseEvent
    PaymentID     string          `json:"payment_id"`
    ProcessedAt   time.Time       `json:"processed_at"`
    FeeAmount     decimal.Decimal `json:"fee_amount"`
    ExchangeRate  decimal.Decimal `json:"exchange_rate,omitempty"`
}

type PaymentFailedEvent struct {
    BaseEvent
    PaymentID string `json:"payment_id"`
    Reason    string `json:"reason"`
    ErrorCode string `json:"error_code"`
}

// Commands
type Command interface {
    GetCommandID() string
    GetAggregateID() string
    GetCommandType() string
}

type BaseCommand struct {
    CommandID   string `json:"command_id"`
    AggregateID string `json:"aggregate_id"`
    CommandType string `json:"command_type"`
}

func (c *BaseCommand) GetCommandID() string   { return c.CommandID }
func (c *BaseCommand) GetAggregateID() string { return c.AggregateID }
func (c *BaseCommand) GetCommandType() string { return c.CommandType }

type InitiatePaymentCommand struct {
    BaseCommand
    PaymentID   string          `json:"payment_id"`
    Amount      decimal.Decimal `json:"amount"`
    Currency    string          `json:"currency"`
    FromAccount string          `json:"from_account"`
    ToAccount   string          `json:"to_account"`
    Reference   string          `json:"reference"`
    UserID      string          `json:"user_id"`
}

type ProcessPaymentCommand struct {
    BaseCommand
    PaymentID string `json:"payment_id"`
}

type CancelPaymentCommand struct {
    BaseCommand
    PaymentID string `json:"payment_id"`
    Reason    string `json:"reason"`
}

// Queries
type Query interface {
    GetQueryID() string
    GetQueryType() string
}

type BaseQuery struct {
    QueryID   string `json:"query_id"`
    QueryType string `json:"query_type"`
}

func (q *BaseQuery) GetQueryID() string   { return q.QueryID }
func (q *BaseQuery) GetQueryType() string { return q.QueryType }

type GetPaymentQuery struct {
    BaseQuery
    PaymentID string `json:"payment_id"`
}

type GetUserPaymentsQuery struct {
    BaseQuery
    UserID string `json:"user_id"`
    Limit  int    `json:"limit"`
    Offset int    `json:"offset"`
}

type GetPaymentAnalyticsQuery struct {
    BaseQuery
    DateFrom time.Time `json:"date_from"`
    DateTo   time.Time `json:"date_to"`
    Currency string    `json:"currency,omitempty"`
}

// Write Model - Payment Aggregate
type PaymentAggregate struct {
    PaymentID   string          `json:"payment_id"`
    Amount      decimal.Decimal `json:"amount"`
    Currency    string          `json:"currency"`
    FromAccount string          `json:"from_account"`
    ToAccount   string          `json:"to_account"`
    Reference   string          `json:"reference"`
    Status      PaymentStatus   `json:"status"`
    CreatedAt   time.Time       `json:"created_at"`
    UpdatedAt   time.Time       `json:"updated_at"`
    Version     int             `json:"version"`

    // Uncommitted events
    uncommittedEvents []Event
}

type PaymentStatus string

const (
    PaymentStatusPending   PaymentStatus = "pending"
    PaymentStatusProcessed PaymentStatus = "processed"
    PaymentStatusFailed    PaymentStatus = "failed"
    PaymentStatusCancelled PaymentStatus = "cancelled"
)

func NewPaymentAggregate() *PaymentAggregate {
    return &PaymentAggregate{
        uncommittedEvents: make([]Event, 0),
    }
}

func (p *PaymentAggregate) InitiatePayment(cmd *InitiatePaymentCommand) error {
    // Business logic validation
    if cmd.Amount.LessThanOrEqual(decimal.Zero) {
        return fmt.Errorf("amount must be positive")
    }

    if cmd.FromAccount == cmd.ToAccount {
        return fmt.Errorf("cannot transfer to same account")
    }

    // Apply the event
    event := &PaymentInitiatedEvent{
        BaseEvent: BaseEvent{
            EventID:     uuid.New().String(),
            AggregateID: cmd.PaymentID,
            EventType:   "PaymentInitiated",
            Timestamp:   time.Now(),
            Version:     p.Version + 1,
        },
        PaymentID:   cmd.PaymentID,
        Amount:      cmd.Amount,
        Currency:    cmd.Currency,
        FromAccount: cmd.FromAccount,
        ToAccount:   cmd.ToAccount,
        Reference:   cmd.Reference,
    }

    p.applyEvent(event)
    return nil
}

func (p *PaymentAggregate) ProcessPayment(cmd *ProcessPaymentCommand) error {
    if p.Status != PaymentStatusPending {
        return fmt.Errorf("payment is not in pending status")
    }

    // Simulate fee calculation
    feeAmount := p.Amount.Mul(decimal.NewFromFloat(0.01)) // 1% fee

    event := &PaymentProcessedEvent{
        BaseEvent: BaseEvent{
            EventID:     uuid.New().String(),
            AggregateID: cmd.PaymentID,
            EventType:   "PaymentProcessed",
            Timestamp:   time.Now(),
            Version:     p.Version + 1,
        },
        PaymentID:   cmd.PaymentID,
        ProcessedAt: time.Now(),
        FeeAmount:   feeAmount,
    }

    p.applyEvent(event)
    return nil
}

func (p *PaymentAggregate) applyEvent(event Event) {
    switch e := event.(type) {
    case *PaymentInitiatedEvent:
        p.PaymentID = e.PaymentID
        p.Amount = e.Amount
        p.Currency = e.Currency
        p.FromAccount = e.FromAccount
        p.ToAccount = e.ToAccount
        p.Reference = e.Reference
        p.Status = PaymentStatusPending
        p.CreatedAt = e.Timestamp
        p.UpdatedAt = e.Timestamp

    case *PaymentProcessedEvent:
        p.Status = PaymentStatusProcessed
        p.UpdatedAt = e.Timestamp

    case *PaymentFailedEvent:
        p.Status = PaymentStatusFailed
        p.UpdatedAt = e.Timestamp
    }

    p.Version = event.GetVersion()
    p.uncommittedEvents = append(p.uncommittedEvents, event)
}

func (p *PaymentAggregate) GetUncommittedEvents() []Event {
    return p.uncommittedEvents
}

func (p *PaymentAggregate) MarkEventsAsCommitted() {
    p.uncommittedEvents = make([]Event, 0)
}

// Read Models
type PaymentView struct {
    PaymentID     string          `json:"payment_id"`
    Amount        decimal.Decimal `json:"amount"`
    Currency      string          `json:"currency"`
    FromAccount   string          `json:"from_account"`
    ToAccount     string          `json:"to_account"`
    Reference     string          `json:"reference"`
    Status        string          `json:"status"`
    CreatedAt     time.Time       `json:"created_at"`
    ProcessedAt   *time.Time      `json:"processed_at,omitempty"`
    FeeAmount     decimal.Decimal `json:"fee_amount"`
    Description   string          `json:"description"`
    Tags          []string        `json:"tags"`
}

type UserPaymentSummary struct {
    UserID           string          `json:"user_id"`
    TotalPayments    int             `json:"total_payments"`
    TotalAmount      decimal.Decimal `json:"total_amount"`
    LastPaymentDate  *time.Time      `json:"last_payment_date,omitempty"`
    FailedPayments   int             `json:"failed_payments"`
    PendingPayments  int             `json:"pending_payments"`
    RecentPayments   []PaymentView   `json:"recent_payments"`
}

type PaymentAnalytics struct {
    TotalPayments     int                        `json:"total_payments"`
    TotalAmount       decimal.Decimal            `json:"total_amount"`
    SuccessRate       float64                    `json:"success_rate"`
    AverageAmount     decimal.Decimal            `json:"average_amount"`
    PaymentsByCurrency map[string]decimal.Decimal `json:"payments_by_currency"`
    PaymentsByStatus  map[string]int             `json:"payments_by_status"`
    DailyVolume       []DailyVolumePoint         `json:"daily_volume"`
}

type DailyVolumePoint struct {
    Date   time.Time       `json:"date"`
    Count  int             `json:"count"`
    Amount decimal.Decimal `json:"amount"`
}

// Command Handlers
type CommandHandler interface {
    Handle(ctx context.Context, cmd Command) error
}

type PaymentCommandHandler struct {
    repository AggregateRepository
    eventBus   EventBus
}

func NewPaymentCommandHandler(repo AggregateRepository, eventBus EventBus) *PaymentCommandHandler {
    return &PaymentCommandHandler{
        repository: repo,
        eventBus:   eventBus,
    }
}

func (h *PaymentCommandHandler) Handle(ctx context.Context, cmd Command) error {
    switch c := cmd.(type) {
    case *InitiatePaymentCommand:
        return h.handleInitiatePayment(ctx, c)
    case *ProcessPaymentCommand:
        return h.handleProcessPayment(ctx, c)
    case *CancelPaymentCommand:
        return h.handleCancelPayment(ctx, c)
    default:
        return fmt.Errorf("unknown command type: %T", cmd)
    }
}

func (h *PaymentCommandHandler) handleInitiatePayment(ctx context.Context, cmd *InitiatePaymentCommand) error {
    aggregate := NewPaymentAggregate()

    if err := aggregate.InitiatePayment(cmd); err != nil {
        return err
    }

    // Save aggregate
    if err := h.repository.Save(ctx, aggregate); err != nil {
        return err
    }

    // Publish events
    for _, event := range aggregate.GetUncommittedEvents() {
        if err := h.eventBus.Publish(ctx, event); err != nil {
            log.Printf("Failed to publish event: %v", err)
        }
    }

    aggregate.MarkEventsAsCommitted()
    return nil
}

func (h *PaymentCommandHandler) handleProcessPayment(ctx context.Context, cmd *ProcessPaymentCommand) error {
    aggregate, err := h.repository.Load(ctx, cmd.PaymentID)
    if err != nil {
        return err
    }

    if err := aggregate.ProcessPayment(cmd); err != nil {
        return err
    }

    if err := h.repository.Save(ctx, aggregate); err != nil {
        return err
    }

    for _, event := range aggregate.GetUncommittedEvents() {
        if err := h.eventBus.Publish(ctx, event); err != nil {
            log.Printf("Failed to publish event: %v", err)
        }
    }

    aggregate.MarkEventsAsCommitted()
    return nil
}

func (h *PaymentCommandHandler) handleCancelPayment(ctx context.Context, cmd *CancelPaymentCommand) error {
    // Implementation similar to processPayment
    return nil
}

// Query Handlers
type QueryHandler interface {
    Handle(ctx context.Context, query Query) (interface{}, error)
}

type PaymentQueryHandler struct {
    readStore ReadStore
}

func NewPaymentQueryHandler(readStore ReadStore) *PaymentQueryHandler {
    return &PaymentQueryHandler{readStore: readStore}
}

func (h *PaymentQueryHandler) Handle(ctx context.Context, query Query) (interface{}, error) {
    switch q := query.(type) {
    case *GetPaymentQuery:
        return h.handleGetPayment(ctx, q)
    case *GetUserPaymentsQuery:
        return h.handleGetUserPayments(ctx, q)
    case *GetPaymentAnalyticsQuery:
        return h.handleGetPaymentAnalytics(ctx, q)
    default:
        return nil, fmt.Errorf("unknown query type: %T", query)
    }
}

func (h *PaymentQueryHandler) handleGetPayment(ctx context.Context, query *GetPaymentQuery) (*PaymentView, error) {
    return h.readStore.GetPayment(ctx, query.PaymentID)
}

func (h *PaymentQueryHandler) handleGetUserPayments(ctx context.Context, query *GetUserPaymentsQuery) (*UserPaymentSummary, error) {
    return h.readStore.GetUserPayments(ctx, query.UserID, query.Limit, query.Offset)
}

func (h *PaymentQueryHandler) handleGetPaymentAnalytics(ctx context.Context, query *GetPaymentAnalyticsQuery) (*PaymentAnalytics, error) {
    return h.readStore.GetPaymentAnalytics(ctx, query.DateFrom, query.DateTo, query.Currency)
}

// Event Handlers for Read Model Updates
type EventHandler interface {
    Handle(ctx context.Context, event Event) error
}

type PaymentViewEventHandler struct {
    readStore ReadStore
}

func NewPaymentViewEventHandler(readStore ReadStore) *PaymentViewEventHandler {
    return &PaymentViewEventHandler{readStore: readStore}
}

func (h *PaymentViewEventHandler) Handle(ctx context.Context, event Event) error {
    switch e := event.(type) {
    case *PaymentInitiatedEvent:
        return h.handlePaymentInitiated(ctx, e)
    case *PaymentProcessedEvent:
        return h.handlePaymentProcessed(ctx, e)
    case *PaymentFailedEvent:
        return h.handlePaymentFailed(ctx, e)
    default:
        return nil // Ignore unknown events
    }
}

func (h *PaymentViewEventHandler) handlePaymentInitiated(ctx context.Context, event *PaymentInitiatedEvent) error {
    view := &PaymentView{
        PaymentID:   event.PaymentID,
        Amount:      event.Amount,
        Currency:    event.Currency,
        FromAccount: event.FromAccount,
        ToAccount:   event.ToAccount,
        Reference:   event.Reference,
        Status:      string(PaymentStatusPending),
        CreatedAt:   event.Timestamp,
        FeeAmount:   decimal.Zero,
        Tags:        make([]string, 0),
    }

    return h.readStore.SavePaymentView(ctx, view)
}

func (h *PaymentViewEventHandler) handlePaymentProcessed(ctx context.Context, event *PaymentProcessedEvent) error {
    return h.readStore.UpdatePaymentStatus(ctx, event.PaymentID, string(PaymentStatusProcessed), &event.ProcessedAt, event.FeeAmount)
}

func (h *PaymentViewEventHandler) handlePaymentFailed(ctx context.Context, event *PaymentFailedEvent) error {
    return h.readStore.UpdatePaymentStatus(ctx, event.PaymentID, string(PaymentStatusFailed), nil, decimal.Zero)
}

// Interfaces
type AggregateRepository interface {
    Save(ctx context.Context, aggregate *PaymentAggregate) error
    Load(ctx context.Context, paymentID string) (*PaymentAggregate, error)
}

type ReadStore interface {
    GetPayment(ctx context.Context, paymentID string) (*PaymentView, error)
    GetUserPayments(ctx context.Context, userID string, limit, offset int) (*UserPaymentSummary, error)
    GetPaymentAnalytics(ctx context.Context, dateFrom, dateTo time.Time, currency string) (*PaymentAnalytics, error)
    SavePaymentView(ctx context.Context, view *PaymentView) error
    UpdatePaymentStatus(ctx context.Context, paymentID, status string, processedAt *time.Time, feeAmount decimal.Decimal) error
}

type EventBus interface {
    Publish(ctx context.Context, event Event) error
    Subscribe(eventType string, handler EventHandler) error
}

// CQRS Bus - Coordinates Commands and Queries
type CQRSBus struct {
    commandHandlers map[string]CommandHandler
    queryHandlers   map[string]QueryHandler
    eventBus        EventBus
    mu              sync.RWMutex
}

func NewCQRSBus(eventBus EventBus) *CQRSBus {
    return &CQRSBus{
        commandHandlers: make(map[string]CommandHandler),
        queryHandlers:   make(map[string]QueryHandler),
        eventBus:        eventBus,
    }
}

func (bus *CQRSBus) RegisterCommandHandler(commandType string, handler CommandHandler) {
    bus.mu.Lock()
    defer bus.mu.Unlock()
    bus.commandHandlers[commandType] = handler
}

func (bus *CQRSBus) RegisterQueryHandler(queryType string, handler QueryHandler) {
    bus.mu.Lock()
    defer bus.mu.Unlock()
    bus.queryHandlers[queryType] = handler
}

func (bus *CQRSBus) ExecuteCommand(ctx context.Context, cmd Command) error {
    bus.mu.RLock()
    handler, exists := bus.commandHandlers[cmd.GetCommandType()]
    bus.mu.RUnlock()

    if !exists {
        return fmt.Errorf("no handler registered for command type: %s", cmd.GetCommandType())
    }

    return handler.Handle(ctx, cmd)
}

func (bus *CQRSBus) ExecuteQuery(ctx context.Context, query Query) (interface{}, error) {
    bus.mu.RLock()
    handler, exists := bus.queryHandlers[query.GetQueryType()]
    bus.mu.RUnlock()

    if !exists {
        return nil, fmt.Errorf("no handler registered for query type: %s", query.GetQueryType())
    }

    return handler.Handle(ctx, query)
}

// In-Memory Implementations (for demo purposes)
type InMemoryEventBus struct {
    handlers map[string][]EventHandler
    mu       sync.RWMutex
}

func NewInMemoryEventBus() *InMemoryEventBus {
    return &InMemoryEventBus{
        handlers: make(map[string][]EventHandler),
    }
}

func (bus *InMemoryEventBus) Publish(ctx context.Context, event Event) error {
    bus.mu.RLock()
    handlers, exists := bus.handlers[event.GetEventType()]
    bus.mu.RUnlock()

    if !exists {
        return nil // No handlers registered
    }

    for _, handler := range handlers {
        if err := handler.Handle(ctx, event); err != nil {
            log.Printf("Error handling event %s: %v", event.GetEventType(), err)
        }
    }

    return nil
}

func (bus *InMemoryEventBus) Subscribe(eventType string, handler EventHandler) error {
    bus.mu.Lock()
    defer bus.mu.Unlock()

    if _, exists := bus.handlers[eventType]; !exists {
        bus.handlers[eventType] = make([]EventHandler, 0)
    }

    bus.handlers[eventType] = append(bus.handlers[eventType], handler)
    return nil
}

type InMemoryAggregateRepository struct {
    aggregates map[string]*PaymentAggregate
    mu         sync.RWMutex
}

func NewInMemoryAggregateRepository() *InMemoryAggregateRepository {
    return &InMemoryAggregateRepository{
        aggregates: make(map[string]*PaymentAggregate),
    }
}

func (repo *InMemoryAggregateRepository) Save(ctx context.Context, aggregate *PaymentAggregate) error {
    repo.mu.Lock()
    defer repo.mu.Unlock()

    // Deep copy to avoid race conditions
    copy := *aggregate
    repo.aggregates[aggregate.PaymentID] = &copy
    return nil
}

func (repo *InMemoryAggregateRepository) Load(ctx context.Context, paymentID string) (*PaymentAggregate, error) {
    repo.mu.RLock()
    defer repo.mu.RUnlock()

    aggregate, exists := repo.aggregates[paymentID]
    if !exists {
        return nil, fmt.Errorf("payment aggregate not found: %s", paymentID)
    }

    // Return a copy
    copy := *aggregate
    return &copy, nil
}

type InMemoryReadStore struct {
    payments map[string]*PaymentView
    mu       sync.RWMutex
}

func NewInMemoryReadStore() *InMemoryReadStore {
    return &InMemoryReadStore{
        payments: make(map[string]*PaymentView),
    }
}

func (store *InMemoryReadStore) GetPayment(ctx context.Context, paymentID string) (*PaymentView, error) {
    store.mu.RLock()
    defer store.mu.RUnlock()

    payment, exists := store.payments[paymentID]
    if !exists {
        return nil, fmt.Errorf("payment not found: %s", paymentID)
    }

    return payment, nil
}

func (store *InMemoryReadStore) GetUserPayments(ctx context.Context, userID string, limit, offset int) (*UserPaymentSummary, error) {
    // Implementation would filter payments by user and create summary
    return &UserPaymentSummary{
        UserID:        userID,
        TotalPayments: 0,
        TotalAmount:   decimal.Zero,
    }, nil
}

func (store *InMemoryReadStore) GetPaymentAnalytics(ctx context.Context, dateFrom, dateTo time.Time, currency string) (*PaymentAnalytics, error) {
    // Implementation would aggregate payment data
    return &PaymentAnalytics{
        TotalPayments:      0,
        TotalAmount:        decimal.Zero,
        SuccessRate:        0.0,
        PaymentsByCurrency: make(map[string]decimal.Decimal),
        PaymentsByStatus:   make(map[string]int),
    }, nil
}

func (store *InMemoryReadStore) SavePaymentView(ctx context.Context, view *PaymentView) error {
    store.mu.Lock()
    defer store.mu.Unlock()

    store.payments[view.PaymentID] = view
    return nil
}

func (store *InMemoryReadStore) UpdatePaymentStatus(ctx context.Context, paymentID, status string, processedAt *time.Time, feeAmount decimal.Decimal) error {
    store.mu.Lock()
    defer store.mu.Unlock()

    payment, exists := store.payments[paymentID]
    if !exists {
        return fmt.Errorf("payment not found: %s", paymentID)
    }

    payment.Status = status
    payment.ProcessedAt = processedAt
    payment.FeeAmount = feeAmount

    return nil
}

// Example usage
func main() {
    fmt.Println("=== CQRS Pattern Demo ===\n")

    // Setup infrastructure
    eventBus := NewInMemoryEventBus()
    aggregateRepo := NewInMemoryAggregateRepository()
    readStore := NewInMemoryReadStore()

    // Setup handlers
    commandHandler := NewPaymentCommandHandler(aggregateRepo, eventBus)
    queryHandler := NewPaymentQueryHandler(readStore)
    eventHandler := NewPaymentViewEventHandler(readStore)

    // Subscribe event handler to events
    eventBus.Subscribe("PaymentInitiated", eventHandler)
    eventBus.Subscribe("PaymentProcessed", eventHandler)
    eventBus.Subscribe("PaymentFailed", eventHandler)

    // Setup CQRS bus
    cqrsBus := NewCQRSBus(eventBus)
    cqrsBus.RegisterCommandHandler("InitiatePayment", commandHandler)
    cqrsBus.RegisterCommandHandler("ProcessPayment", commandHandler)
    cqrsBus.RegisterQueryHandler("GetPayment", queryHandler)
    cqrsBus.RegisterQueryHandler("GetUserPayments", queryHandler)

    ctx := context.Background()

    // Example 1: Initiate a payment (Command)
    fmt.Println("1. Initiating Payment (Command)")
    paymentID := uuid.New().String()
    initiateCmd := &InitiatePaymentCommand{
        BaseCommand: BaseCommand{
            CommandID:   uuid.New().String(),
            AggregateID: paymentID,
            CommandType: "InitiatePayment",
        },
        PaymentID:   paymentID,
        Amount:      decimal.NewFromFloat(100.50),
        Currency:    "USD",
        FromAccount: "ACC_001",
        ToAccount:   "ACC_002",
        Reference:   "Payment for services",
        UserID:      "USER_123",
    }

    if err := cqrsBus.ExecuteCommand(ctx, initiateCmd); err != nil {
        log.Fatalf("Failed to initiate payment: %v", err)
    }
    fmt.Printf("Payment initiated: %s\n", paymentID)

    // Small delay to ensure event processing
    time.Sleep(100 * time.Millisecond)

    // Example 2: Query the payment (Query)
    fmt.Println("\n2. Querying Payment (Query)")
    getPaymentQuery := &GetPaymentQuery{
        BaseQuery: BaseQuery{
            QueryID:   uuid.New().String(),
            QueryType: "GetPayment",
        },
        PaymentID: paymentID,
    }

    result, err := cqrsBus.ExecuteQuery(ctx, getPaymentQuery)
    if err != nil {
        log.Fatalf("Failed to query payment: %v", err)
    }

    paymentView := result.(*PaymentView)
    fmt.Printf("Payment found: %+v\n", paymentView)

    // Example 3: Process the payment (Command)
    fmt.Println("\n3. Processing Payment (Command)")
    processCmd := &ProcessPaymentCommand{
        BaseCommand: BaseCommand{
            CommandID:   uuid.New().String(),
            AggregateID: paymentID,
            CommandType: "ProcessPayment",
        },
        PaymentID: paymentID,
    }

    if err := cqrsBus.ExecuteCommand(ctx, processCmd); err != nil {
        log.Fatalf("Failed to process payment: %v", err)
    }
    fmt.Printf("Payment processed: %s\n", paymentID)

    // Small delay to ensure event processing
    time.Sleep(100 * time.Millisecond)

    // Example 4: Query the updated payment
    fmt.Println("\n4. Querying Updated Payment (Query)")
    result, err = cqrsBus.ExecuteQuery(ctx, getPaymentQuery)
    if err != nil {
        log.Fatalf("Failed to query payment: %v", err)
    }

    updatedPaymentView := result.(*PaymentView)
    fmt.Printf("Updated payment: %+v\n", updatedPaymentView)

    // Example 5: Multiple payments for analytics
    fmt.Println("\n5. Creating Multiple Payments for Analytics")
    for i := 0; i < 5; i++ {
        pid := uuid.New().String()
        cmd := &InitiatePaymentCommand{
            BaseCommand: BaseCommand{
                CommandID:   uuid.New().String(),
                AggregateID: pid,
                CommandType: "InitiatePayment",
            },
            PaymentID:   pid,
            Amount:      decimal.NewFromFloat(float64(50 + i*25)),
            Currency:    "USD",
            FromAccount: fmt.Sprintf("ACC_%03d", i+10),
            ToAccount:   fmt.Sprintf("ACC_%03d", i+20),
            Reference:   fmt.Sprintf("Payment #%d", i+1),
            UserID:      "USER_123",
        }

        if err := cqrsBus.ExecuteCommand(ctx, cmd); err != nil {
            log.Printf("Failed to create payment %d: %v", i+1, err)
        }
    }

    fmt.Printf("Created 5 additional payments\n")

    fmt.Println("\n=== CQRS Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Simple CQRS (Shared Database)**

```go
type SimpleCQRS struct {
    writeDB *sql.DB // Single database
    readDB  *sql.DB // Same database, different connection
}
```

2. **CQRS with Separate Databases**

```go
type AdvancedCQRS struct {
    writeDB *sql.DB     // Write-optimized database
    readDB  *mongodb.DB // Read-optimized database
}
```

3. **Event-Sourced CQRS**

```go
type EventSourcedCQRS struct {
    eventStore EventStore
    readStore  ReadStore
    projector  EventProjector
}
```

4. **Microservices CQRS**

```go
type DistributedCQRS struct {
    commandService *CommandService
    queryService   *QueryService
    eventBus       DistributedEventBus
}
```

### Trade-offs

**Pros:**

- **Independent Scaling**: Scale read and write operations separately
- **Optimized Models**: Different models for different use cases
- **Performance**: Optimize read and write operations independently
- **Flexibility**: Easy to add new query models without affecting writes
- **Separation of Concerns**: Clear separation between commands and queries

**Cons:**

- **Complexity**: More complex than simple CRUD
- **Eventual Consistency**: Read models may be slightly behind
- **Infrastructure**: Requires event bus and potentially multiple databases
- **Learning Curve**: Team needs to understand CQRS concepts
- **Debugging**: More complex to debug distributed flows

**When to Choose CQRS vs Alternatives:**

| Scenario                        | Pattern                 | Reason                   |
| ------------------------------- | ----------------------- | ------------------------ |
| Complex read/write requirements | CQRS                    | Independent optimization |
| Simple CRUD operations          | Repository              | Less complexity          |
| Event-driven architecture       | Event Sourcing + CQRS   | Natural fit              |
| High read volume                | CQRS with read replicas | Optimized queries        |
| Microservices                   | CQRS                    | Service independence     |

## Testable Example

```go
package main

import (
    "context"
    "testing"
    "time"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/shopspring/decimal"
    "github.com/google/uuid"
)

func TestPaymentAggregate_InitiatePayment(t *testing.T) {
    aggregate := NewPaymentAggregate()

    cmd := &InitiatePaymentCommand{
        BaseCommand: BaseCommand{
            CommandID:   uuid.New().String(),
            AggregateID: "PAY_001",
            CommandType: "InitiatePayment",
        },
        PaymentID:   "PAY_001",
        Amount:      decimal.NewFromFloat(100.0),
        Currency:    "USD",
        FromAccount: "ACC_001",
        ToAccount:   "ACC_002",
        Reference:   "Test payment",
    }

    err := aggregate.InitiatePayment(cmd)

    assert.NoError(t, err)
    assert.Equal(t, "PAY_001", aggregate.PaymentID)
    assert.Equal(t, decimal.NewFromFloat(100.0), aggregate.Amount)
    assert.Equal(t, "USD", aggregate.Currency)
    assert.Equal(t, PaymentStatusPending, aggregate.Status)
    assert.Len(t, aggregate.GetUncommittedEvents(), 1)

    event := aggregate.GetUncommittedEvents()[0]
    assert.Equal(t, "PaymentInitiated", event.GetEventType())
}

func TestPaymentAggregate_ProcessPayment(t *testing.T) {
    aggregate := NewPaymentAggregate()

    // First initiate payment
    initiateCmd := &InitiatePaymentCommand{
        BaseCommand: BaseCommand{
            CommandID:   uuid.New().String(),
            AggregateID: "PAY_001",
            CommandType: "InitiatePayment",
        },
        PaymentID:   "PAY_001",
        Amount:      decimal.NewFromFloat(100.0),
        Currency:    "USD",
        FromAccount: "ACC_001",
        ToAccount:   "ACC_002",
        Reference:   "Test payment",
    }

    err := aggregate.InitiatePayment(initiateCmd)
    require.NoError(t, err)

    // Then process payment
    processCmd := &ProcessPaymentCommand{
        BaseCommand: BaseCommand{
            CommandID:   uuid.New().String(),
            AggregateID: "PAY_001",
            CommandType: "ProcessPayment",
        },
        PaymentID: "PAY_001",
    }

    err = aggregate.ProcessPayment(processCmd)

    assert.NoError(t, err)
    assert.Equal(t, PaymentStatusProcessed, aggregate.Status)
    assert.Len(t, aggregate.GetUncommittedEvents(), 2)

    events := aggregate.GetUncommittedEvents()
    assert.Equal(t, "PaymentInitiated", events[0].GetEventType())
    assert.Equal(t, "PaymentProcessed", events[1].GetEventType())
}

func TestPaymentAggregate_ValidationErrors(t *testing.T) {
    aggregate := NewPaymentAggregate()

    tests := []struct {
        name    string
        cmd     *InitiatePaymentCommand
        wantErr string
    }{
        {
            name: "negative amount",
            cmd: &InitiatePaymentCommand{
                PaymentID:   "PAY_001",
                Amount:      decimal.NewFromFloat(-100.0),
                Currency:    "USD",
                FromAccount: "ACC_001",
                ToAccount:   "ACC_002",
            },
            wantErr: "amount must be positive",
        },
        {
            name: "same account transfer",
            cmd: &InitiatePaymentCommand{
                PaymentID:   "PAY_001",
                Amount:      decimal.NewFromFloat(100.0),
                Currency:    "USD",
                FromAccount: "ACC_001",
                ToAccount:   "ACC_001",
            },
            wantErr: "cannot transfer to same account",
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := aggregate.InitiatePayment(tt.cmd)
            assert.Error(t, err)
            assert.Contains(t, err.Error(), tt.wantErr)
        })
    }
}

func TestCQRSBus_CommandExecution(t *testing.T) {
    eventBus := NewInMemoryEventBus()
    repo := NewInMemoryAggregateRepository()
    commandHandler := NewPaymentCommandHandler(repo, eventBus)

    cqrsBus := NewCQRSBus(eventBus)
    cqrsBus.RegisterCommandHandler("InitiatePayment", commandHandler)

    ctx := context.Background()

    cmd := &InitiatePaymentCommand{
        BaseCommand: BaseCommand{
            CommandID:   uuid.New().String(),
            AggregateID: "PAY_001",
            CommandType: "InitiatePayment",
        },
        PaymentID:   "PAY_001",
        Amount:      decimal.NewFromFloat(100.0),
        Currency:    "USD",
        FromAccount: "ACC_001",
        ToAccount:   "ACC_002",
        Reference:   "Test payment",
    }

    err := cqrsBus.ExecuteCommand(ctx, cmd)
    assert.NoError(t, err)

    // Verify aggregate was saved
    aggregate, err := repo.Load(ctx, "PAY_001")
    assert.NoError(t, err)
    assert.Equal(t, "PAY_001", aggregate.PaymentID)
    assert.Equal(t, PaymentStatusPending, aggregate.Status)
}

func TestCQRSBus_QueryExecution(t *testing.T) {
    readStore := NewInMemoryReadStore()
    queryHandler := NewPaymentQueryHandler(readStore)

    cqrsBus := NewCQRSBus(nil)
    cqrsBus.RegisterQueryHandler("GetPayment", queryHandler)

    ctx := context.Background()

    // Setup test data
    paymentView := &PaymentView{
        PaymentID:   "PAY_001",
        Amount:      decimal.NewFromFloat(100.0),
        Currency:    "USD",
        FromAccount: "ACC_001",
        ToAccount:   "ACC_002",
        Status:      string(PaymentStatusPending),
        CreatedAt:   time.Now(),
    }

    err := readStore.SavePaymentView(ctx, paymentView)
    require.NoError(t, err)

    // Execute query
    query := &GetPaymentQuery{
        BaseQuery: BaseQuery{
            QueryID:   uuid.New().String(),
            QueryType: "GetPayment",
        },
        PaymentID: "PAY_001",
    }

    result, err := cqrsBus.ExecuteQuery(ctx, query)
    assert.NoError(t, err)

    resultView := result.(*PaymentView)
    assert.Equal(t, "PAY_001", resultView.PaymentID)
    assert.Equal(t, decimal.NewFromFloat(100.0), resultView.Amount)
    assert.Equal(t, "USD", resultView.Currency)
}

func TestEventHandling(t *testing.T) {
    readStore := NewInMemoryReadStore()
    eventHandler := NewPaymentViewEventHandler(readStore)

    ctx := context.Background()

    // Handle PaymentInitiated event
    event := &PaymentInitiatedEvent{
        BaseEvent: BaseEvent{
            EventID:     uuid.New().String(),
            AggregateID: "PAY_001",
            EventType:   "PaymentInitiated",
            Timestamp:   time.Now(),
            Version:     1,
        },
        PaymentID:   "PAY_001",
        Amount:      decimal.NewFromFloat(100.0),
        Currency:    "USD",
        FromAccount: "ACC_001",
        ToAccount:   "ACC_002",
        Reference:   "Test payment",
    }

    err := eventHandler.Handle(ctx, event)
    assert.NoError(t, err)

    // Verify read model was updated
    paymentView, err := readStore.GetPayment(ctx, "PAY_001")
    assert.NoError(t, err)
    assert.Equal(t, "PAY_001", paymentView.PaymentID)
    assert.Equal(t, string(PaymentStatusPending), paymentView.Status)

    // Handle PaymentProcessed event
    processedEvent := &PaymentProcessedEvent{
        BaseEvent: BaseEvent{
            EventID:     uuid.New().String(),
            AggregateID: "PAY_001",
            EventType:   "PaymentProcessed",
            Timestamp:   time.Now(),
            Version:     2,
        },
        PaymentID:   "PAY_001",
        ProcessedAt: time.Now(),
        FeeAmount:   decimal.NewFromFloat(1.0),
    }

    err = eventHandler.Handle(ctx, processedEvent)
    assert.NoError(t, err)

    // Verify status was updated
    updatedView, err := readStore.GetPayment(ctx, "PAY_001")
    assert.NoError(t, err)
    assert.Equal(t, string(PaymentStatusProcessed), updatedView.Status)
    assert.Equal(t, decimal.NewFromFloat(1.0), updatedView.FeeAmount)
    assert.NotNil(t, updatedView.ProcessedAt)
}

func TestEventualConsistency(t *testing.T) {
    eventBus := NewInMemoryEventBus()
    repo := NewInMemoryAggregateRepository()
    readStore := NewInMemoryReadStore()

    commandHandler := NewPaymentCommandHandler(repo, eventBus)
    queryHandler := NewPaymentQueryHandler(readStore)
    eventHandler := NewPaymentViewEventHandler(readStore)

    // Subscribe event handler
    eventBus.Subscribe("PaymentInitiated", eventHandler)

    cqrsBus := NewCQRSBus(eventBus)
    cqrsBus.RegisterCommandHandler("InitiatePayment", commandHandler)
    cqrsBus.RegisterQueryHandler("GetPayment", queryHandler)

    ctx := context.Background()

    // Execute command
    cmd := &InitiatePaymentCommand{
        BaseCommand: BaseCommand{
            CommandID:   uuid.New().String(),
            AggregateID: "PAY_001",
            CommandType: "InitiatePayment",
        },
        PaymentID:   "PAY_001",
        Amount:      decimal.NewFromFloat(100.0),
        Currency:    "USD",
        FromAccount: "ACC_001",
        ToAccount:   "ACC_002",
        Reference:   "Test payment",
    }

    err := cqrsBus.ExecuteCommand(ctx, cmd)
    assert.NoError(t, err)

    // Small delay for event processing
    time.Sleep(10 * time.Millisecond)

    // Query should return the payment
    query := &GetPaymentQuery{
        BaseQuery: BaseQuery{
            QueryID:   uuid.New().String(),
            QueryType: "GetPayment",
        },
        PaymentID: "PAY_001",
    }

    result, err := cqrsBus.ExecuteQuery(ctx, query)
    assert.NoError(t, err)

    paymentView := result.(*PaymentView)
    assert.Equal(t, "PAY_001", paymentView.PaymentID)
    assert.Equal(t, string(PaymentStatusPending), paymentView.Status)
}

func BenchmarkCommandExecution(b *testing.B) {
    eventBus := NewInMemoryEventBus()
    repo := NewInMemoryAggregateRepository()
    commandHandler := NewPaymentCommandHandler(repo, eventBus)

    cqrsBus := NewCQRSBus(eventBus)
    cqrsBus.RegisterCommandHandler("InitiatePayment", commandHandler)

    ctx := context.Background()

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        cmd := &InitiatePaymentCommand{
            BaseCommand: BaseCommand{
                CommandID:   uuid.New().String(),
                AggregateID: fmt.Sprintf("PAY_%d", i),
                CommandType: "InitiatePayment",
            },
            PaymentID:   fmt.Sprintf("PAY_%d", i),
            Amount:      decimal.NewFromFloat(100.0),
            Currency:    "USD",
            FromAccount: "ACC_001",
            ToAccount:   "ACC_002",
            Reference:   "Benchmark payment",
        }

        err := cqrsBus.ExecuteCommand(ctx, cmd)
        if err != nil {
            b.Fatal(err)
        }
    }
}

func BenchmarkQueryExecution(b *testing.B) {
    readStore := NewInMemoryReadStore()
    queryHandler := NewPaymentQueryHandler(readStore)

    cqrsBus := NewCQRSBus(nil)
    cqrsBus.RegisterQueryHandler("GetPayment", queryHandler)

    ctx := context.Background()

    // Setup test data
    for i := 0; i < 1000; i++ {
        paymentView := &PaymentView{
            PaymentID:   fmt.Sprintf("PAY_%d", i),
            Amount:      decimal.NewFromFloat(100.0),
            Currency:    "USD",
            Status:      string(PaymentStatusPending),
            CreatedAt:   time.Now(),
        }
        readStore.SavePaymentView(ctx, paymentView)
    }

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        query := &GetPaymentQuery{
            BaseQuery: BaseQuery{
                QueryID:   uuid.New().String(),
                QueryType: "GetPayment",
            },
            PaymentID: fmt.Sprintf("PAY_%d", i%1000),
        }

        _, err := cqrsBus.ExecuteQuery(ctx, query)
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

## Integration Tips

### 1. Database Integration

```go
// MySQL for write model
type MySQLAggregateRepository struct {
    db *sql.DB
}

func (r *MySQLAggregateRepository) Save(ctx context.Context, aggregate *PaymentAggregate) error {
    tx, err := r.db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }
    defer tx.Rollback()

    // Save aggregate
    _, err = tx.ExecContext(ctx, `
        INSERT INTO payments (payment_id, amount, currency, status, version, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON DUPLICATE KEY UPDATE
        amount = VALUES(amount),
        status = VALUES(status),
        version = VALUES(version),
        updated_at = NOW()
    `, aggregate.PaymentID, aggregate.Amount, aggregate.Currency, aggregate.Status, aggregate.Version, aggregate.CreatedAt)

    if err != nil {
        return err
    }

    // Save events
    for _, event := range aggregate.GetUncommittedEvents() {
        eventData, _ := json.Marshal(event)
        _, err = tx.ExecContext(ctx, `
            INSERT INTO events (event_id, aggregate_id, event_type, event_data, version, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        `, event.GetEventID(), event.GetAggregateID(), event.GetEventType(), eventData, event.GetVersion(), event.GetTimestamp())

        if err != nil {
            return err
        }
    }

    return tx.Commit()
}

// MongoDB for read model
type MongoReadStore struct {
    db *mongo.Database
}

func (s *MongoReadStore) SavePaymentView(ctx context.Context, view *PaymentView) error {
    collection := s.db.Collection("payment_views")

    _, err := collection.ReplaceOne(ctx,
        bson.M{"payment_id": view.PaymentID},
        view,
        options.Replace().SetUpsert(true),
    )

    return err
}
```

### 2. Kafka Integration

```go
type KafkaEventBus struct {
    producer *kafka.Writer
    consumer *kafka.Reader
    handlers map[string][]EventHandler
}

func (k *KafkaEventBus) Publish(ctx context.Context, event Event) error {
    eventData, err := json.Marshal(event)
    if err != nil {
        return err
    }

    message := kafka.Message{
        Key:   []byte(event.GetAggregateID()),
        Value: eventData,
        Headers: []kafka.Header{
            {Key: "event_type", Value: []byte(event.GetEventType())},
            {Key: "aggregate_id", Value: []byte(event.GetAggregateID())},
        },
    }

    return k.producer.WriteMessages(ctx, message)
}

func (k *KafkaEventBus) StartConsumer(ctx context.Context) {
    for {
        message, err := k.consumer.ReadMessage(ctx)
        if err != nil {
            log.Printf("Error reading message: %v", err)
            continue
        }

        // Get event type from headers
        var eventType string
        for _, header := range message.Headers {
            if header.Key == "event_type" {
                eventType = string(header.Value)
                break
            }
        }

        // Handle event
        if handlers, exists := k.handlers[eventType]; exists {
            for _, handler := range handlers {
                // Deserialize event and handle
                go k.handleEvent(handler, message.Value, eventType)
            }
        }
    }
}
```

### 3. HTTP API Integration

```go
type PaymentAPI struct {
    cqrsBus *CQRSBus
    router  *gin.Engine
}

func (api *PaymentAPI) setupRoutes() {
    api.router.POST("/payments", api.createPayment)
    api.router.GET("/payments/:id", api.getPayment)
    api.router.POST("/payments/:id/process", api.processPayment)
    api.router.GET("/users/:userId/payments", api.getUserPayments)
}

func (api *PaymentAPI) createPayment(c *gin.Context) {
    var req struct {
        Amount      float64 `json:"amount" binding:"required,gt=0"`
        Currency    string  `json:"currency" binding:"required,len=3"`
        FromAccount string  `json:"from_account" binding:"required"`
        ToAccount   string  `json:"to_account" binding:"required"`
        Reference   string  `json:"reference"`
        UserID      string  `json:"user_id" binding:"required"`
    }

    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }

    paymentID := uuid.New().String()
    cmd := &InitiatePaymentCommand{
        BaseCommand: BaseCommand{
            CommandID:   uuid.New().String(),
            AggregateID: paymentID,
            CommandType: "InitiatePayment",
        },
        PaymentID:   paymentID,
        Amount:      decimal.NewFromFloat(req.Amount),
        Currency:    req.Currency,
        FromAccount: req.FromAccount,
        ToAccount:   req.ToAccount,
        Reference:   req.Reference,
        UserID:      req.UserID,
    }

    if err := api.cqrsBus.ExecuteCommand(c.Request.Context(), cmd); err != nil {
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }

    c.JSON(201, gin.H{"payment_id": paymentID, "status": "initiated"})
}

func (api *PaymentAPI) getPayment(c *gin.Context) {
    paymentID := c.Param("id")

    query := &GetPaymentQuery{
        BaseQuery: BaseQuery{
            QueryID:   uuid.New().String(),
            QueryType: "GetPayment",
        },
        PaymentID: paymentID,
    }

    result, err := api.cqrsBus.ExecuteQuery(c.Request.Context(), query)
    if err != nil {
        c.JSON(404, gin.H{"error": "payment not found"})
        return
    }

    c.JSON(200, result)
}
```

### 4. Circuit Breaker Integration

```go
type ResilientCQRSBus struct {
    *CQRSBus
    commandCircuitBreaker *CircuitBreaker
    queryCircuitBreaker   *CircuitBreaker
}

func (r *ResilientCQRSBus) ExecuteCommand(ctx context.Context, cmd Command) error {
    return r.commandCircuitBreaker.Execute(func() error {
        return r.CQRSBus.ExecuteCommand(ctx, cmd)
    })
}

func (r *ResilientCQRSBus) ExecuteQuery(ctx context.Context, query Query) (interface{}, error) {
    var result interface{}
    var err error

    circuitErr := r.queryCircuitBreaker.Execute(func() error {
        result, err = r.CQRSBus.ExecuteQuery(ctx, query)
        return err
    })

    if circuitErr != nil {
        return nil, circuitErr
    }

    return result, err
}
```

## Common Interview Questions

### 1. **What problems does CQRS solve?**

**Answer:**
CQRS solves several key problems:

1. **Different Read/Write Requirements**: Read and write operations often have different performance, consistency, and data model requirements
2. **Scalability Bottlenecks**: Traditional systems can't optimize reads and writes independently
3. **Complex Queries vs Simple Commands**: Reporting queries are often complex while commands are simple operations
4. **Data Model Mismatch**: The same data model rarely serves both operational and analytical needs optimally

**Example:**

```go
// Write model - optimized for transactions
type PaymentCommand struct {
    PaymentID string
    Amount    decimal.Decimal
    // ... minimal fields
}

// Read model - optimized for queries
type PaymentAnalyticsView struct {
    PaymentID       string
    UserProfile     UserProfile
    RiskScore      float64
    SimilarPayments []PaymentSummary
    // ... rich denormalized data
}
```

### 2. **How do you handle eventual consistency in CQRS?**

**Answer:**
Eventual consistency is managed through several strategies:

1. **Event-Driven Updates**: Read models are updated asynchronously via events
2. **Compensation Mechanisms**: Handle out-of-order or failed event processing
3. **User Experience**: Design UI to handle stale data gracefully
4. **Monitoring**: Track read model lag and health

```go
type EventProcessor struct {
    lastProcessedEventID string
    retryPolicy         RetryPolicy
    deadLetterQueue     DeadLetterQueue
}

func (p *EventProcessor) ProcessEvent(event Event) error {
    if err := p.updateReadModel(event); err != nil {
        // Retry with exponential backoff
        return p.retryPolicy.Execute(func() error {
            return p.updateReadModel(event)
        })
    }

    p.lastProcessedEventID = event.GetEventID()
    return nil
}
```

### 3. **When should you NOT use CQRS?**

**Answer:**
Avoid CQRS when:

1. **Simple CRUD**: Application is mostly simple create/read/update/delete operations
2. **Small Team**: Team lacks distributed systems experience
3. **Strong Consistency Required**: Cannot tolerate any data inconsistency
4. **Simple Domain**: Business logic is straightforward without complex rules

**Simple CRUD Example (avoid CQRS):**

```go
type UserService struct {
    userRepo UserRepository
}

// Simple operations don't need CQRS
func (s *UserService) CreateUser(user *User) error {
    return s.userRepo.Save(user)
}

func (s *UserService) GetUser(id string) (*User, error) {
    return s.userRepo.FindByID(id)
}
```

### 4. **How do you implement CQRS with microservices?**

**Answer:**
CQRS in microservices involves:

1. **Service Separation**: Separate command and query services
2. **Event Bus**: Distributed event bus (Kafka, RabbitMQ)
3. **Database per Service**: Each service has its own database
4. **API Gateway**: Route commands and queries appropriately

```go
// Command Service
type PaymentCommandService struct {
    commandBus CQRSBus
    eventBus   DistributedEventBus
}

// Query Service
type PaymentQueryService struct {
    queryBus  CQRSBus
    readStore ReadStore
}

// API Gateway routing
func (g *APIGateway) Route(req *http.Request) {
    if req.Method == "POST" || req.Method == "PUT" || req.Method == "DELETE" {
        // Route to command service
        g.commandService.Handle(req)
    } else {
        // Route to query service
        g.queryService.Handle(req)
    }
}
```

### 5. **How do you handle transactions in CQRS?**

**Answer:**
Transactions in CQRS are handled differently:

1. **Command Side**: Traditional ACID transactions within aggregates
2. **Cross-Aggregate**: Eventual consistency via saga pattern
3. **Read Side**: Eventually consistent updates
4. **Compensation**: Compensating actions for failures

```go
// Single aggregate transaction
func (h *PaymentCommandHandler) Handle(ctx context.Context, cmd *TransferCommand) error {
    return h.repository.WithTransaction(ctx, func(tx Transaction) error {
        // Load aggregate
        payment, err := tx.LoadPayment(cmd.PaymentID)
        if err != nil {
            return err
        }

        // Execute command
        if err := payment.Transfer(cmd); err != nil {
            return err
        }

        // Save aggregate
        return tx.SavePayment(payment)
    })
}

// Cross-aggregate with saga
type TransferSaga struct {
    steps []SagaStep
}

func (s *TransferSaga) Execute(ctx context.Context) error {
    for i, step := range s.steps {
        if err := step.Execute(ctx); err != nil {
            // Compensate previous steps
            s.compensate(ctx, i-1)
            return err
        }
    }
    return nil
}
```
