# Event Sourcing Pattern

## Pattern Name & Intent

**Event Sourcing** is an architectural pattern where state changes are stored as a sequence of events. Instead of storing current state, the system stores all events that led to the current state.

**Key Intent:**
- Store state changes as immutable events
- Rebuild application state by replaying events
- Provide complete audit trail of all changes
- Enable temporal queries and time travel debugging
- Support eventual consistency in distributed systems
- Enable complex business intelligence and analytics

## When to Use

**Use Event Sourcing when:**

1. **Audit Requirements**: Need complete audit trail of all changes
2. **Complex Domain**: Business logic with complex state transitions
3. **Temporal Queries**: Need to query historical state
4. **Undo/Replay**: Requirement to undo operations or replay scenarios
5. **Analytics**: Need detailed analytics on user behavior
6. **Microservices**: Building event-driven microservices
7. **High Scalability**: Need to scale reads independently from writes

**Don't use when:**
- Simple CRUD applications
- Low latency requirements for reads
- Complex querying is primary use case
- Team lacks expertise in event-driven architecture

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Transaction Event Sourcing
```go
// Payment events
type PaymentEvent interface {
    GetEventID() string
    GetAggregateID() string
    GetEventType() string
    GetTimestamp() time.Time
    GetVersion() int64
    GetEventData() interface{}
}

type BasePaymentEvent struct {
    EventID     string
    AggregateID string
    EventType   string
    Timestamp   time.Time
    Version     int64
    Metadata    map[string]interface{}
}

func (bpe *BasePaymentEvent) GetEventID() string     { return bpe.EventID }
func (bpe *BasePaymentEvent) GetAggregateID() string { return bpe.AggregateID }
func (bpe *BasePaymentEvent) GetEventType() string   { return bpe.EventType }
func (bpe *BasePaymentEvent) GetTimestamp() time.Time { return bpe.Timestamp }
func (bpe *BasePaymentEvent) GetVersion() int64      { return bpe.Version }

// Specific payment events
type PaymentInitiatedEvent struct {
    BasePaymentEvent
    Amount        decimal.Decimal `json:"amount"`
    Currency      string         `json:"currency"`
    CustomerID    string         `json:"customer_id"`
    PaymentMethod string         `json:"payment_method"`
    Description   string         `json:"description"`
}

func (pie *PaymentInitiatedEvent) GetEventData() interface{} {
    return struct {
        Amount        decimal.Decimal `json:"amount"`
        Currency      string         `json:"currency"`
        CustomerID    string         `json:"customer_id"`
        PaymentMethod string         `json:"payment_method"`
        Description   string         `json:"description"`
    }{
        Amount:        pie.Amount,
        Currency:      pie.Currency,
        CustomerID:    pie.CustomerID,
        PaymentMethod: pie.PaymentMethod,
        Description:   pie.Description,
    }
}

type PaymentValidatedEvent struct {
    BasePaymentEvent
    ValidationType string `json:"validation_type"`
    RiskScore      float64 `json:"risk_score"`
}

func (pve *PaymentValidatedEvent) GetEventData() interface{} {
    return struct {
        ValidationType string  `json:"validation_type"`
        RiskScore      float64 `json:"risk_score"`
    }{
        ValidationType: pve.ValidationType,
        RiskScore:      pve.RiskScore,
    }
}

type PaymentProcessedEvent struct {
    BasePaymentEvent
    TransactionID string    `json:"transaction_id"`
    Gateway       string    `json:"gateway"`
    ProcessedAt   time.Time `json:"processed_at"`
    Fee           decimal.Decimal `json:"fee"`
}

func (ppe *PaymentProcessedEvent) GetEventData() interface{} {
    return struct {
        TransactionID string          `json:"transaction_id"`
        Gateway       string          `json:"gateway"`
        ProcessedAt   time.Time       `json:"processed_at"`
        Fee           decimal.Decimal `json:"fee"`
    }{
        TransactionID: ppe.TransactionID,
        Gateway:       ppe.Gateway,
        ProcessedAt:   ppe.ProcessedAt,
        Fee:           ppe.Fee,
    }
}

type PaymentFailedEvent struct {
    BasePaymentEvent
    ErrorCode    string `json:"error_code"`
    ErrorMessage string `json:"error_message"`
    FailureType  string `json:"failure_type"`
}

func (pfe *PaymentFailedEvent) GetEventData() interface{} {
    return struct {
        ErrorCode    string `json:"error_code"`
        ErrorMessage string `json:"error_message"`
        FailureType  string `json:"failure_type"`
    }{
        ErrorCode:    pfe.ErrorCode,
        ErrorMessage: pfe.ErrorMessage,
        FailureType:  pfe.FailureType,
    }
}

// Payment aggregate
type PaymentAggregate struct {
    ID            string
    Amount        decimal.Decimal
    Currency      string
    CustomerID    string
    PaymentMethod string
    Status        string
    TransactionID string
    Gateway       string
    RiskScore     float64
    Fee           decimal.Decimal
    CreatedAt     time.Time
    ProcessedAt   time.Time
    Version       int64
    
    // Event sourcing specific
    uncommittedEvents []PaymentEvent
    eventHistory      []PaymentEvent
}

func NewPaymentAggregate(id string) *PaymentAggregate {
    return &PaymentAggregate{
        ID:                id,
        Status:            "UNKNOWN",
        uncommittedEvents: make([]PaymentEvent, 0),
        eventHistory:      make([]PaymentEvent, 0),
    }
}

// Command methods (generate events)
func (pa *PaymentAggregate) InitiatePayment(amount decimal.Decimal, currency, customerID, paymentMethod, description string) error {
    if pa.Status != "UNKNOWN" {
        return fmt.Errorf("payment already initiated")
    }
    
    event := &PaymentInitiatedEvent{
        BasePaymentEvent: BasePaymentEvent{
            EventID:     generateEventID(),
            AggregateID: pa.ID,
            EventType:   "PaymentInitiated",
            Timestamp:   time.Now(),
            Version:     pa.Version + 1,
            Metadata:    make(map[string]interface{}),
        },
        Amount:        amount,
        Currency:      currency,
        CustomerID:    customerID,
        PaymentMethod: paymentMethod,
        Description:   description,
    }
    
    pa.applyEvent(event)
    pa.uncommittedEvents = append(pa.uncommittedEvents, event)
    
    return nil
}

func (pa *PaymentAggregate) ValidatePayment(validationType string, riskScore float64) error {
    if pa.Status != "INITIATED" {
        return fmt.Errorf("payment not in initiated state")
    }
    
    event := &PaymentValidatedEvent{
        BasePaymentEvent: BasePaymentEvent{
            EventID:     generateEventID(),
            AggregateID: pa.ID,
            EventType:   "PaymentValidated",
            Timestamp:   time.Now(),
            Version:     pa.Version + 1,
        },
        ValidationType: validationType,
        RiskScore:      riskScore,
    }
    
    pa.applyEvent(event)
    pa.uncommittedEvents = append(pa.uncommittedEvents, event)
    
    return nil
}

func (pa *PaymentAggregate) ProcessPayment(transactionID, gateway string, fee decimal.Decimal) error {
    if pa.Status != "VALIDATED" {
        return fmt.Errorf("payment not validated")
    }
    
    event := &PaymentProcessedEvent{
        BasePaymentEvent: BasePaymentEvent{
            EventID:     generateEventID(),
            AggregateID: pa.ID,
            EventType:   "PaymentProcessed",
            Timestamp:   time.Now(),
            Version:     pa.Version + 1,
        },
        TransactionID: transactionID,
        Gateway:       gateway,
        ProcessedAt:   time.Now(),
        Fee:           fee,
    }
    
    pa.applyEvent(event)
    pa.uncommittedEvents = append(pa.uncommittedEvents, event)
    
    return nil
}

func (pa *PaymentAggregate) FailPayment(errorCode, errorMessage, failureType string) error {
    if pa.Status == "PROCESSED" {
        return fmt.Errorf("cannot fail processed payment")
    }
    
    event := &PaymentFailedEvent{
        BasePaymentEvent: BasePaymentEvent{
            EventID:     generateEventID(),
            AggregateID: pa.ID,
            EventType:   "PaymentFailed",
            Timestamp:   time.Now(),
            Version:     pa.Version + 1,
        },
        ErrorCode:    errorCode,
        ErrorMessage: errorMessage,
        FailureType:  failureType,
    }
    
    pa.applyEvent(event)
    pa.uncommittedEvents = append(pa.uncommittedEvents, event)
    
    return nil
}

// Event application (state changes)
func (pa *PaymentAggregate) applyEvent(event PaymentEvent) {
    switch e := event.(type) {
    case *PaymentInitiatedEvent:
        pa.Amount = e.Amount
        pa.Currency = e.Currency
        pa.CustomerID = e.CustomerID
        pa.PaymentMethod = e.PaymentMethod
        pa.Status = "INITIATED"
        pa.CreatedAt = e.Timestamp
        
    case *PaymentValidatedEvent:
        pa.RiskScore = e.RiskScore
        pa.Status = "VALIDATED"
        
    case *PaymentProcessedEvent:
        pa.TransactionID = e.TransactionID
        pa.Gateway = e.Gateway
        pa.ProcessedAt = e.ProcessedAt
        pa.Fee = e.Fee
        pa.Status = "PROCESSED"
        
    case *PaymentFailedEvent:
        pa.Status = "FAILED"
    }
    
    pa.Version = event.GetVersion()
    pa.eventHistory = append(pa.eventHistory, event)
}

// Event sourcing methods
func (pa *PaymentAggregate) GetUncommittedEvents() []PaymentEvent {
    return pa.uncommittedEvents
}

func (pa *PaymentAggregate) MarkEventsAsCommitted() {
    pa.uncommittedEvents = make([]PaymentEvent, 0)
}

func (pa *PaymentAggregate) LoadFromHistory(events []PaymentEvent) {
    for _, event := range events {
        pa.applyEvent(event)
    }
    pa.uncommittedEvents = make([]PaymentEvent, 0)
}

func (pa *PaymentAggregate) GetStateAtVersion(version int64) *PaymentAggregate {
    temp := NewPaymentAggregate(pa.ID)
    
    for _, event := range pa.eventHistory {
        if event.GetVersion() <= version {
            temp.applyEvent(event)
        } else {
            break
        }
    }
    
    return temp
}

// Event store interface
type EventStore interface {
    SaveEvents(ctx context.Context, aggregateID string, events []PaymentEvent, expectedVersion int64) error
    GetEvents(ctx context.Context, aggregateID string) ([]PaymentEvent, error)
    GetEventsFromVersion(ctx context.Context, aggregateID string, fromVersion int64) ([]PaymentEvent, error)
    GetAllEvents(ctx context.Context, fromTimestamp time.Time) ([]PaymentEvent, error)
}

// In-memory event store implementation
type InMemoryEventStore struct {
    events map[string][]PaymentEvent
    mu     sync.RWMutex
    logger *zap.Logger
}

func NewInMemoryEventStore(logger *zap.Logger) *InMemoryEventStore {
    return &InMemoryEventStore{
        events: make(map[string][]PaymentEvent),
        logger: logger,
    }
}

func (imes *InMemoryEventStore) SaveEvents(ctx context.Context, aggregateID string, events []PaymentEvent, expectedVersion int64) error {
    imes.mu.Lock()
    defer imes.mu.Unlock()
    
    existingEvents := imes.events[aggregateID]
    
    // Optimistic concurrency check
    currentVersion := int64(0)
    if len(existingEvents) > 0 {
        currentVersion = existingEvents[len(existingEvents)-1].GetVersion()
    }
    
    if currentVersion != expectedVersion {
        return fmt.Errorf("concurrency conflict: expected version %d, current version %d", 
            expectedVersion, currentVersion)
    }
    
    // Append new events
    imes.events[aggregateID] = append(existingEvents, events...)
    
    imes.logger.Debug("Events saved", 
        zap.String("aggregate_id", aggregateID),
        zap.Int("event_count", len(events)),
        zap.Int64("new_version", events[len(events)-1].GetVersion()))
    
    return nil
}

func (imes *InMemoryEventStore) GetEvents(ctx context.Context, aggregateID string) ([]PaymentEvent, error) {
    imes.mu.RLock()
    defer imes.mu.RUnlock()
    
    events := imes.events[aggregateID]
    if events == nil {
        return make([]PaymentEvent, 0), nil
    }
    
    // Return copy to prevent external modification
    result := make([]PaymentEvent, len(events))
    copy(result, events)
    
    return result, nil
}

func (imes *InMemoryEventStore) GetEventsFromVersion(ctx context.Context, aggregateID string, fromVersion int64) ([]PaymentEvent, error) {
    imes.mu.RLock()
    defer imes.mu.RUnlock()
    
    events := imes.events[aggregateID]
    if events == nil {
        return make([]PaymentEvent, 0), nil
    }
    
    result := make([]PaymentEvent, 0)
    for _, event := range events {
        if event.GetVersion() >= fromVersion {
            result = append(result, event)
        }
    }
    
    return result, nil
}

func (imes *InMemoryEventStore) GetAllEvents(ctx context.Context, fromTimestamp time.Time) ([]PaymentEvent, error) {
    imes.mu.RLock()
    defer imes.mu.RUnlock()
    
    result := make([]PaymentEvent, 0)
    for _, events := range imes.events {
        for _, event := range events {
            if event.GetTimestamp().After(fromTimestamp) || event.GetTimestamp().Equal(fromTimestamp) {
                result = append(result, event)
            }
        }
    }
    
    // Sort by timestamp
    sort.Slice(result, func(i, j int) bool {
        return result[i].GetTimestamp().Before(result[j].GetTimestamp())
    })
    
    return result, nil
}

// Repository for event sourced aggregates
type PaymentRepository struct {
    eventStore EventStore
    logger     *zap.Logger
}

func NewPaymentRepository(eventStore EventStore, logger *zap.Logger) *PaymentRepository {
    return &PaymentRepository{
        eventStore: eventStore,
        logger:     logger,
    }
}

func (pr *PaymentRepository) Save(ctx context.Context, aggregate *PaymentAggregate) error {
    uncommittedEvents := aggregate.GetUncommittedEvents()
    if len(uncommittedEvents) == 0 {
        return nil
    }
    
    expectedVersion := aggregate.Version - int64(len(uncommittedEvents))
    
    if err := pr.eventStore.SaveEvents(ctx, aggregate.ID, uncommittedEvents, expectedVersion); err != nil {
        return fmt.Errorf("failed to save events: %w", err)
    }
    
    aggregate.MarkEventsAsCommitted()
    
    pr.logger.Debug("Payment aggregate saved", 
        zap.String("aggregate_id", aggregate.ID),
        zap.Int("events_saved", len(uncommittedEvents)))
    
    return nil
}

func (pr *PaymentRepository) GetByID(ctx context.Context, id string) (*PaymentAggregate, error) {
    events, err := pr.eventStore.GetEvents(ctx, id)
    if err != nil {
        return nil, fmt.Errorf("failed to get events: %w", err)
    }
    
    if len(events) == 0 {
        return nil, fmt.Errorf("payment aggregate not found: %s", id)
    }
    
    aggregate := NewPaymentAggregate(id)
    aggregate.LoadFromHistory(events)
    
    pr.logger.Debug("Payment aggregate loaded", 
        zap.String("aggregate_id", id),
        zap.Int("events_loaded", len(events)),
        zap.Int64("version", aggregate.Version))
    
    return aggregate, nil
}

func (pr *PaymentRepository) GetAtVersion(ctx context.Context, id string, version int64) (*PaymentAggregate, error) {
    aggregate, err := pr.GetByID(ctx, id)
    if err != nil {
        return nil, err
    }
    
    return aggregate.GetStateAtVersion(version), nil
}

// Event projections for read models
type PaymentProjection struct {
    ID            string
    CustomerID    string
    Amount        decimal.Decimal
    Currency      string
    Status        string
    PaymentMethod string
    TransactionID string
    Gateway       string
    RiskScore     float64
    Fee           decimal.Decimal
    CreatedAt     time.Time
    ProcessedAt   time.Time
    LastUpdated   time.Time
}

type PaymentProjectionStore interface {
    Save(ctx context.Context, projection *PaymentProjection) error
    GetByID(ctx context.Context, id string) (*PaymentProjection, error)
    GetByCustomerID(ctx context.Context, customerID string) ([]*PaymentProjection, error)
    GetByStatus(ctx context.Context, status string) ([]*PaymentProjection, error)
}

// Event handlers for projections
type PaymentProjectionHandler struct {
    projectionStore PaymentProjectionStore
    logger         *zap.Logger
}

func NewPaymentProjectionHandler(store PaymentProjectionStore, logger *zap.Logger) *PaymentProjectionHandler {
    return &PaymentProjectionHandler{
        projectionStore: store,
        logger:         logger,
    }
}

func (pph *PaymentProjectionHandler) HandleEvent(ctx context.Context, event PaymentEvent) error {
    switch e := event.(type) {
    case *PaymentInitiatedEvent:
        return pph.handlePaymentInitiated(ctx, e)
    case *PaymentValidatedEvent:
        return pph.handlePaymentValidated(ctx, e)
    case *PaymentProcessedEvent:
        return pph.handlePaymentProcessed(ctx, e)
    case *PaymentFailedEvent:
        return pph.handlePaymentFailed(ctx, e)
    default:
        pph.logger.Warn("Unknown event type", zap.String("event_type", event.GetEventType()))
        return nil
    }
}

func (pph *PaymentProjectionHandler) handlePaymentInitiated(ctx context.Context, event *PaymentInitiatedEvent) error {
    projection := &PaymentProjection{
        ID:            event.AggregateID,
        CustomerID:    event.CustomerID,
        Amount:        event.Amount,
        Currency:      event.Currency,
        Status:        "INITIATED",
        PaymentMethod: event.PaymentMethod,
        CreatedAt:     event.Timestamp,
        LastUpdated:   event.Timestamp,
    }
    
    return pph.projectionStore.Save(ctx, projection)
}

func (pph *PaymentProjectionHandler) handlePaymentValidated(ctx context.Context, event *PaymentValidatedEvent) error {
    projection, err := pph.projectionStore.GetByID(ctx, event.AggregateID)
    if err != nil {
        return err
    }
    
    projection.Status = "VALIDATED"
    projection.RiskScore = event.RiskScore
    projection.LastUpdated = event.Timestamp
    
    return pph.projectionStore.Save(ctx, projection)
}

func (pph *PaymentProjectionHandler) handlePaymentProcessed(ctx context.Context, event *PaymentProcessedEvent) error {
    projection, err := pph.projectionStore.GetByID(ctx, event.AggregateID)
    if err != nil {
        return err
    }
    
    projection.Status = "PROCESSED"
    projection.TransactionID = event.TransactionID
    projection.Gateway = event.Gateway
    projection.Fee = event.Fee
    projection.ProcessedAt = event.ProcessedAt
    projection.LastUpdated = event.Timestamp
    
    return pph.projectionStore.Save(ctx, projection)
}

func (pph *PaymentProjectionHandler) handlePaymentFailed(ctx context.Context, event *PaymentFailedEvent) error {
    projection, err := pph.projectionStore.GetByID(ctx, event.AggregateID)
    if err != nil {
        return err
    }
    
    projection.Status = "FAILED"
    projection.LastUpdated = event.Timestamp
    
    return pph.projectionStore.Save(ctx, projection)
}

// Helper functions
func generateEventID() string {
    return fmt.Sprintf("evt_%d_%s", time.Now().UnixNano(), randomString(8))
}

func randomString(length int) string {
    const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    result := make([]byte, length)
    for i := range result {
        result[i] = chars[rand.Intn(len(chars))]
    }
    return string(result)
}
```

## Go Implementation

```go
package main

import (
    "context"
    "fmt"
    "sort"
    "time"
    "go.uber.org/zap"
    "github.com/shopspring/decimal"
)

// Simple example of event sourcing for a bank account
// Demonstrates basic concepts of event sourcing pattern

// Account events
type AccountEvent interface {
    GetEventID() string
    GetAccountID() string
    GetEventType() string
    GetTimestamp() time.Time
    GetAmount() decimal.Decimal
}

type AccountOpenedEvent struct {
    EventID     string
    AccountID   string
    CustomerID  string
    InitialBalance decimal.Decimal
    Timestamp   time.Time
}

func (e *AccountOpenedEvent) GetEventID() string { return e.EventID }
func (e *AccountOpenedEvent) GetAccountID() string { return e.AccountID }
func (e *AccountOpenedEvent) GetEventType() string { return "AccountOpened" }
func (e *AccountOpenedEvent) GetTimestamp() time.Time { return e.Timestamp }
func (e *AccountOpenedEvent) GetAmount() decimal.Decimal { return e.InitialBalance }

type MoneyDepositedEvent struct {
    EventID   string
    AccountID string
    Amount    decimal.Decimal
    Timestamp time.Time
    Reference string
}

func (e *MoneyDepositedEvent) GetEventID() string { return e.EventID }
func (e *MoneyDepositedEvent) GetAccountID() string { return e.AccountID }
func (e *MoneyDepositedEvent) GetEventType() string { return "MoneyDeposited" }
func (e *MoneyDepositedEvent) GetTimestamp() time.Time { return e.Timestamp }
func (e *MoneyDepositedEvent) GetAmount() decimal.Decimal { return e.Amount }

type MoneyWithdrawnEvent struct {
    EventID   string
    AccountID string
    Amount    decimal.Decimal
    Timestamp time.Time
    Reference string
}

func (e *MoneyWithdrawnEvent) GetEventID() string { return e.EventID }
func (e *MoneyWithdrawnEvent) GetAccountID() string { return e.AccountID }
func (e *MoneyWithdrawnEvent) GetEventType() string { return "MoneyWithdrawn" }
func (e *MoneyWithdrawnEvent) GetTimestamp() time.Time { return e.Timestamp }
func (e *MoneyWithdrawnEvent) GetAmount() decimal.Decimal { return e.Amount }

// Account aggregate
type Account struct {
    ID                string
    CustomerID        string
    Balance           decimal.Decimal
    OpenedAt          time.Time
    LastTransactionAt time.Time
    IsActive          bool
    
    events []AccountEvent
    logger *zap.Logger
}

func NewAccount(id, customerID string, logger *zap.Logger) *Account {
    return &Account{
        ID:         id,
        CustomerID: customerID,
        Balance:    decimal.Zero,
        IsActive:   false,
        events:     make([]AccountEvent, 0),
        logger:     logger,
    }
}

// Commands
func (a *Account) OpenAccount(initialBalance decimal.Decimal) error {
    if a.IsActive {
        return fmt.Errorf("account already opened")
    }
    
    event := &AccountOpenedEvent{
        EventID:        generateEventID(),
        AccountID:      a.ID,
        CustomerID:     a.CustomerID,
        InitialBalance: initialBalance,
        Timestamp:      time.Now(),
    }
    
    a.applyEvent(event)
    a.events = append(a.events, event)
    
    return nil
}

func (a *Account) Deposit(amount decimal.Decimal, reference string) error {
    if !a.IsActive {
        return fmt.Errorf("account not active")
    }
    
    if amount.LessThanOrEqual(decimal.Zero) {
        return fmt.Errorf("deposit amount must be positive")
    }
    
    event := &MoneyDepositedEvent{
        EventID:   generateEventID(),
        AccountID: a.ID,
        Amount:    amount,
        Timestamp: time.Now(),
        Reference: reference,
    }
    
    a.applyEvent(event)
    a.events = append(a.events, event)
    
    return nil
}

func (a *Account) Withdraw(amount decimal.Decimal, reference string) error {
    if !a.IsActive {
        return fmt.Errorf("account not active")
    }
    
    if amount.LessThanOrEqual(decimal.Zero) {
        return fmt.Errorf("withdrawal amount must be positive")
    }
    
    if a.Balance.LessThan(amount) {
        return fmt.Errorf("insufficient funds: balance %s, requested %s", 
            a.Balance.String(), amount.String())
    }
    
    event := &MoneyWithdrawnEvent{
        EventID:   generateEventID(),
        AccountID: a.ID,
        Amount:    amount,
        Timestamp: time.Now(),
        Reference: reference,
    }
    
    a.applyEvent(event)
    a.events = append(a.events, event)
    
    return nil
}

// Event application
func (a *Account) applyEvent(event AccountEvent) {
    switch e := event.(type) {
    case *AccountOpenedEvent:
        a.Balance = e.InitialBalance
        a.OpenedAt = e.Timestamp
        a.LastTransactionAt = e.Timestamp
        a.IsActive = true
        
    case *MoneyDepositedEvent:
        a.Balance = a.Balance.Add(e.Amount)
        a.LastTransactionAt = e.Timestamp
        
    case *MoneyWithdrawnEvent:
        a.Balance = a.Balance.Sub(e.Amount)
        a.LastTransactionAt = e.Timestamp
    }
    
    a.logger.Debug("Event applied", 
        zap.String("account_id", a.ID),
        zap.String("event_type", event.GetEventType()),
        zap.String("new_balance", a.Balance.String()))
}

// Event sourcing methods
func (a *Account) GetUncommittedEvents() []AccountEvent {
    return a.events
}

func (a *Account) MarkEventsAsCommitted() {
    a.events = make([]AccountEvent, 0)
}

func (a *Account) LoadFromHistory(events []AccountEvent) {
    for _, event := range events {
        a.applyEvent(event)
    }
    a.events = make([]AccountEvent, 0)
}

// Simple event store
type SimpleEventStore struct {
    events map[string][]AccountEvent
    logger *zap.Logger
}

func NewSimpleEventStore(logger *zap.Logger) *SimpleEventStore {
    return &SimpleEventStore{
        events: make(map[string][]AccountEvent),
        logger: logger,
    }
}

func (ses *SimpleEventStore) SaveEvents(ctx context.Context, accountID string, events []AccountEvent) error {
    if len(events) == 0 {
        return nil
    }
    
    existingEvents := ses.events[accountID]
    if existingEvents == nil {
        existingEvents = make([]AccountEvent, 0)
    }
    
    ses.events[accountID] = append(existingEvents, events...)
    
    ses.logger.Info("Events saved", 
        zap.String("account_id", accountID),
        zap.Int("event_count", len(events)))
    
    return nil
}

func (ses *SimpleEventStore) GetEvents(ctx context.Context, accountID string) ([]AccountEvent, error) {
    events := ses.events[accountID]
    if events == nil {
        return make([]AccountEvent, 0), nil
    }
    
    // Return copy
    result := make([]AccountEvent, len(events))
    copy(result, events)
    
    return result, nil
}

// Account repository
type AccountRepository struct {
    eventStore *SimpleEventStore
    logger     *zap.Logger
}

func NewAccountRepository(eventStore *SimpleEventStore, logger *zap.Logger) *AccountRepository {
    return &AccountRepository{
        eventStore: eventStore,
        logger:     logger,
    }
}

func (ar *AccountRepository) Save(ctx context.Context, account *Account) error {
    uncommittedEvents := account.GetUncommittedEvents()
    if len(uncommittedEvents) == 0 {
        return nil
    }
    
    if err := ar.eventStore.SaveEvents(ctx, account.ID, uncommittedEvents); err != nil {
        return fmt.Errorf("failed to save events: %w", err)
    }
    
    account.MarkEventsAsCommitted()
    
    ar.logger.Debug("Account saved", 
        zap.String("account_id", account.ID),
        zap.Int("events_saved", len(uncommittedEvents)))
    
    return nil
}

func (ar *AccountRepository) GetByID(ctx context.Context, accountID, customerID string) (*Account, error) {
    events, err := ar.eventStore.GetEvents(ctx, accountID)
    if err != nil {
        return nil, fmt.Errorf("failed to get events: %w", err)
    }
    
    account := NewAccount(accountID, customerID, ar.logger)
    account.LoadFromHistory(events)
    
    ar.logger.Debug("Account loaded", 
        zap.String("account_id", accountID),
        zap.Int("events_loaded", len(events)))
    
    return account, nil
}

// Example usage
func main() {
    fmt.Println("=== Event Sourcing Pattern Demo ===\n")
    
    // Create logger
    logger, _ := zap.NewDevelopment()
    defer logger.Sync()
    
    // Create event store and repository
    eventStore := NewSimpleEventStore(logger)
    accountRepo := NewAccountRepository(eventStore, logger)
    
    ctx := context.Background()
    
    // Example 1: Create and operate on account
    fmt.Println("=== Account Operations ===")
    
    account := NewAccount("acc_001", "customer_123", logger)
    
    // Open account with initial deposit
    if err := account.OpenAccount(decimal.NewFromInt(1000)); err != nil {
        fmt.Printf("Failed to open account: %v\n", err)
        return
    }
    
    fmt.Printf("Account opened with balance: $%s\n", account.Balance.String())
    
    // Make some transactions
    err := account.Deposit(decimal.NewFromInt(500), "salary_deposit")
    if err != nil {
        fmt.Printf("Deposit failed: %v\n", err)
        return
    }
    
    fmt.Printf("After deposit: $%s\n", account.Balance.String())
    
    err = account.Withdraw(decimal.NewFromInt(200), "atm_withdrawal")
    if err != nil {
        fmt.Printf("Withdrawal failed: %v\n", err)
        return
    }
    
    fmt.Printf("After withdrawal: $%s\n", account.Balance.String())
    
    // Save account (persist events)
    if err := accountRepo.Save(ctx, account); err != nil {
        fmt.Printf("Failed to save account: %v\n", err)
        return
    }
    
    fmt.Printf("Account saved with %d events\n", len(account.GetUncommittedEvents()))
    
    // Example 2: Load account from events
    fmt.Println("\n=== Loading Account from Events ===")
    
    loadedAccount, err := accountRepo.GetByID(ctx, "acc_001", "customer_123")
    if err != nil {
        fmt.Printf("Failed to load account: %v\n", err)
        return
    }
    
    fmt.Printf("Loaded account balance: $%s\n", loadedAccount.Balance.String())
    fmt.Printf("Account opened at: %s\n", loadedAccount.OpenedAt.Format("2006-01-02 15:04:05"))
    fmt.Printf("Last transaction at: %s\n", loadedAccount.LastTransactionAt.Format("2006-01-02 15:04:05"))
    fmt.Printf("Account active: %t\n", loadedAccount.IsActive)
    
    // Example 3: Show event history
    fmt.Println("\n=== Event History ===")
    
    events, err := eventStore.GetEvents(ctx, "acc_001")
    if err != nil {
        fmt.Printf("Failed to get events: %v\n", err)
        return
    }
    
    fmt.Printf("Total events: %d\n", len(events))
    
    for i, event := range events {
        fmt.Printf("%d. %s at %s - Amount: $%s\n", 
            i+1,
            event.GetEventType(),
            event.GetTimestamp().Format("15:04:05"),
            event.GetAmount().String())
    }
    
    // Example 4: Demonstrate event replay for different time points
    fmt.Println("\n=== Event Replay ===")
    
    replayAccount := NewAccount("acc_001", "customer_123", logger)
    
    // Replay only first two events
    if len(events) >= 2 {
        replayAccount.LoadFromHistory(events[:2])
        fmt.Printf("Balance after first 2 events: $%s\n", replayAccount.Balance.String())
    }
    
    // Replay all events
    replayAccount = NewAccount("acc_001", "customer_123", logger)
    replayAccount.LoadFromHistory(events)
    fmt.Printf("Balance after all events: $%s\n", replayAccount.Balance.String())
    
    // Example 5: Show benefits of event sourcing
    fmt.Println("\n=== Event Sourcing Benefits ===")
    fmt.Printf("1. Complete audit trail: %d events recorded\n", len(events))
    fmt.Printf("2. Point-in-time reconstruction: Can rebuild state at any point\n")
    fmt.Printf("3. Temporal queries: Can analyze account behavior over time\n")
    fmt.Printf("4. Event replay: Can replay events for testing or analysis\n")
    fmt.Printf("5. Immutable history: Events cannot be changed, only new events added\n")
    
    // Example 6: Demonstrate error handling
    fmt.Println("\n=== Error Handling ===")
    
    // Try to withdraw more than balance
    err = loadedAccount.Withdraw(decimal.NewFromInt(2000), "large_withdrawal")
    if err != nil {
        fmt.Printf("Expected error: %v\n", err)
    }
    
    // Try to operate on unopened account
    newAccount := NewAccount("acc_002", "customer_456", logger)
    err = newAccount.Deposit(decimal.NewFromInt(100), "test_deposit")
    if err != nil {
        fmt.Printf("Expected error: %v\n", err)
    }
    
    fmt.Println("\n=== Event Sourcing Pattern Demo Complete ===")
}

func generateEventID() string {
    return fmt.Sprintf("evt_%d", time.Now().UnixNano())
}
```

## Variants & Trade-offs

### Variants

1. **CQRS Integration**
```go
type CommandHandler interface {
    Handle(ctx context.Context, command interface{}) error
}

type QueryHandler interface {
    Handle(ctx context.Context, query interface{}) (interface{}, error)
}

type PaymentCommandHandler struct {
    repository PaymentRepository
}

func (pch *PaymentCommandHandler) Handle(ctx context.Context, command interface{}) error {
    switch cmd := command.(type) {
    case *InitiatePaymentCommand:
        return pch.handleInitiatePayment(ctx, cmd)
    case *ProcessPaymentCommand:
        return pch.handleProcessPayment(ctx, cmd)
    }
    return fmt.Errorf("unknown command type")
}
```

2. **Snapshots**
```go
type Snapshot struct {
    AggregateID   string
    AggregateType string
    Version       int64
    Timestamp     time.Time
    Data          interface{}
}

type SnapshotStore interface {
    SaveSnapshot(ctx context.Context, snapshot *Snapshot) error
    GetSnapshot(ctx context.Context, aggregateID string) (*Snapshot, error)
}

func (pr *PaymentRepository) GetByIDWithSnapshot(ctx context.Context, id string) (*PaymentAggregate, error) {
    snapshot, _ := pr.snapshotStore.GetSnapshot(ctx, id)
    
    var fromVersion int64 = 0
    var aggregate *PaymentAggregate
    
    if snapshot != nil {
        aggregate = snapshot.Data.(*PaymentAggregate)
        fromVersion = snapshot.Version + 1
    } else {
        aggregate = NewPaymentAggregate(id)
    }
    
    events, err := pr.eventStore.GetEventsFromVersion(ctx, id, fromVersion)
    if err != nil {
        return nil, err
    }
    
    aggregate.LoadFromHistory(events)
    return aggregate, nil
}
```

### Trade-offs

**Pros:**
- **Complete Audit Trail**: Every change is recorded as an event
- **Temporal Queries**: Can query state at any point in time
- **Scalability**: Reads and writes can be scaled independently
- **Debugging**: Easy to replay events to understand system behavior
- **Integration**: Natural fit for event-driven architectures

**Cons:**
- **Complexity**: Adds significant complexity to the system
- **Performance**: Read operations require event replay
- **Storage**: Events accumulate over time
- **Learning Curve**: Requires different thinking about data modeling
- **Eventual Consistency**: Read models may lag behind events

## Integration Tips

### 1. **Saga Pattern Integration**
```go
type PaymentSaga struct {
    events []PaymentEvent
    state  string
}

func (ps *PaymentSaga) Handle(event PaymentEvent) error {
    ps.events = append(ps.events, event)
    
    switch event.GetEventType() {
    case "PaymentInitiated":
        return ps.validatePayment(event)
    case "PaymentValidated":
        return ps.processPayment(event)
    }
    return nil
}
```

### 2. **Message Bus Integration**
```go
type EventPublisher interface {
    Publish(ctx context.Context, event PaymentEvent) error
}

func (pr *PaymentRepository) Save(ctx context.Context, aggregate *PaymentAggregate) error {
    events := aggregate.GetUncommittedEvents()
    
    if err := pr.eventStore.SaveEvents(ctx, aggregate.ID, events, aggregate.Version); err != nil {
        return err
    }
    
    // Publish events to message bus
    for _, event := range events {
        if err := pr.publisher.Publish(ctx, event); err != nil {
            pr.logger.Error("Failed to publish event", zap.Error(err))
        }
    }
    
    aggregate.MarkEventsAsCommitted()
    return nil
}
```

## Common Interview Questions

### 1. **How does Event Sourcing differ from traditional CRUD?**

**Traditional CRUD:**
- Stores current state only
- Updates modify existing data
- No history of changes
- Simple querying

**Event Sourcing:**
- Stores all events leading to current state
- Appends events, never updates
- Complete history preserved
- State derived from events

### 2. **How do you handle schema evolution in Event Sourcing?**

**Versioned Events:**
```go
type PaymentEventV2 struct {
    Version int `json:"version"`
    // ... event data
}

func (pe *PaymentEventV2) Migrate() PaymentEvent {
    if pe.Version == 1 {
        // Convert V1 to current format
    }
    return pe
}
```

### 3. **When should you use snapshots?**

Use snapshots when:
- Event replay becomes slow
- Large number of events per aggregate
- Complex event application logic
- Performance requirements for reads

### 4. **How do you query Event Sourced data?**

**Projections/Read Models:**
```go
type PaymentReadModel struct {
    ID       string
    Status   string
    Amount   decimal.Decimal
    Customer string
}

type PaymentProjector struct {
    store PaymentReadModelStore
}

func (pp *PaymentProjector) Project(event PaymentEvent) error {
    switch e := event.(type) {
    case *PaymentInitiatedEvent:
        readModel := &PaymentReadModel{
            ID:       e.AggregateID,
            Status:   "INITIATED",
            Amount:   e.Amount,
            Customer: e.CustomerID,
        }
        return pp.store.Save(readModel)
    }
    return nil
}
```

### 5. **How do you handle eventual consistency?**

**Strategies:**
- Use read models that may lag behind events
- Implement compensating actions for failures
- Design UX to handle eventual consistency
- Use optimistic UI updates with rollback capability
