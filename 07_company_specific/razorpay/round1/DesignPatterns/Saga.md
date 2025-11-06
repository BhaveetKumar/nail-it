---
# Auto-generated front matter
Title: Saga
LastUpdated: 2025-11-06T20:45:58.518587
Tags: []
Status: draft
---

# Saga Pattern

## Pattern Name & Intent

**Saga** is a design pattern for managing data consistency across microservices in distributed transaction scenarios. It provides a way to manage long-running transactions by breaking them into a series of smaller, local transactions with compensating actions.

**Key Intent:**

- Maintain data consistency across distributed services
- Handle long-running business processes
- Provide fault tolerance and recovery mechanisms
- Avoid distributed locks and two-phase commits
- Enable rollback through compensating transactions

## When to Use

**Use Saga when:**

1. **Distributed Transactions**: Need to maintain consistency across multiple services
2. **Long-Running Processes**: Business processes span multiple steps and services
3. **Microservices Architecture**: Services are distributed and autonomous
4. **Eventual Consistency**: Can tolerate temporary inconsistency
5. **Complex Business Workflows**: Multi-step processes with dependencies
6. **High Availability**: Cannot afford to lock resources for long periods

**Don't use when:**

- Single service operations (use local transactions)
- Strong consistency is absolutely required
- Simple operations without compensation logic
- System has low complexity and few services

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Processing Saga

```go
// Multi-step payment process across services
type PaymentSaga struct {
    PaymentID     string
    UserID        string
    Amount        decimal.Decimal
    MerchantID    string
    Steps         []SagaStep
    Compensations []CompensationStep
}

// Steps: Validate User -> Reserve Funds -> Process Payment -> Update Merchant Balance -> Send Notification
func NewPaymentSaga(paymentID, userID, merchantID string, amount decimal.Decimal) *PaymentSaga {
    return &PaymentSaga{
        PaymentID:  paymentID,
        UserID:     userID,
        MerchantID: merchantID,
        Amount:     amount,
        Steps: []SagaStep{
            &ValidateUserStep{},
            &ReserveFundsStep{},
            &ProcessPaymentStep{},
            &UpdateMerchantBalanceStep{},
            &SendNotificationStep{},
        },
        Compensations: []CompensationStep{
            &ReleaseReservationCompensation{},
            &RevertPaymentCompensation{},
            &RevertMerchantBalanceCompensation{},
        },
    }
}
```

### 2. Account Opening Saga

```go
// Multi-service account opening process
type AccountOpeningSaga struct {
    ApplicationID string
    CustomerInfo  CustomerInfo
    Steps         []SagaStep
}

// Steps: KYC Verification -> Credit Check -> Create Account -> Issue Cards -> Setup Services
func NewAccountOpeningSaga(applicationID string, customer CustomerInfo) *AccountOpeningSaga {
    return &AccountOpeningSaga{
        ApplicationID: applicationID,
        CustomerInfo:  customer,
        Steps: []SagaStep{
            &KYCVerificationStep{},
            &CreditCheckStep{},
            &CreateAccountStep{},
            &IssueCardsStep{},
            &SetupServicesStep{},
        },
    }
}
```

### 3. Trade Settlement Saga

```go
// Multi-party trade settlement
type TradeSettlementSaga struct {
    TradeID      string
    BuyerID      string
    SellerID     string
    Instrument   string
    Quantity     int64
    Price        decimal.Decimal
    Steps        []SagaStep
}

// Steps: Validate Trade -> Reserve Securities -> Reserve Cash -> Execute Transfer -> Update Positions -> Clear Trade
func NewTradeSettlementSaga(trade TradeInfo) *TradeSettlementSaga {
    return &TradeSettlementSaga{
        TradeID:    trade.TradeID,
        BuyerID:    trade.BuyerID,
        SellerID:   trade.SellerID,
        Instrument: trade.Instrument,
        Quantity:   trade.Quantity,
        Price:      trade.Price,
        Steps: []SagaStep{
            &ValidateTradeStep{},
            &ReserveSecuritiesStep{},
            &ReserveCashStep{},
            &ExecuteTransferStep{},
            &UpdatePositionsStep{},
            &ClearTradeStep{},
        },
    }
}
```

### 4. Loan Approval Saga

```go
// Multi-stage loan approval process
type LoanApprovalSaga struct {
    ApplicationID string
    ApplicantID   string
    LoanAmount    decimal.Decimal
    LoanType      string
    Steps         []SagaStep
}

// Steps: Document Verification -> Credit Assessment -> Collateral Evaluation -> Risk Analysis -> Final Approval -> Disbursement
func NewLoanApprovalSaga(app LoanApplication) *LoanApprovalSaga {
    return &LoanApprovalSaga{
        ApplicationID: app.ApplicationID,
        ApplicantID:   app.ApplicantID,
        LoanAmount:    app.LoanAmount,
        LoanType:      app.LoanType,
        Steps: []SagaStep{
            &DocumentVerificationStep{},
            &CreditAssessmentStep{},
            &CollateralEvaluationStep{},
            &RiskAnalysisStep{},
            &FinalApprovalStep{},
            &DisbursementStep{},
        },
    }
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

// Saga States
type SagaState string

const (
    SagaStateStarted     SagaState = "started"
    SagaStateInProgress  SagaState = "in_progress"
    SagaStateCompleted   SagaState = "completed"
    SagaStateFailed      SagaState = "failed"
    SagaStateCompensating SagaState = "compensating"
    SagaStateCompensated SagaState = "compensated"
)

// Step States
type StepState string

const (
    StepStatePending     StepState = "pending"
    StepStateExecuting   StepState = "executing"
    StepStateCompleted   StepState = "completed"
    StepStateFailed      StepState = "failed"
    StepStateCompensating StepState = "compensating"
    StepStateCompensated StepState = "compensated"
)

// Saga Context holds shared data
type SagaContext struct {
    SagaID    string                 `json:"saga_id"`
    Data      map[string]interface{} `json:"data"`
    CreatedAt time.Time              `json:"created_at"`
    UpdatedAt time.Time              `json:"updated_at"`
    mu        sync.RWMutex
}

func NewSagaContext(sagaID string) *SagaContext {
    return &SagaContext{
        SagaID:    sagaID,
        Data:      make(map[string]interface{}),
        CreatedAt: time.Now(),
        UpdatedAt: time.Now(),
    }
}

func (ctx *SagaContext) Set(key string, value interface{}) {
    ctx.mu.Lock()
    defer ctx.mu.Unlock()
    ctx.Data[key] = value
    ctx.UpdatedAt = time.Now()
}

func (ctx *SagaContext) Get(key string) (interface{}, bool) {
    ctx.mu.RLock()
    defer ctx.mu.RUnlock()
    value, exists := ctx.Data[key]
    return value, exists
}

func (ctx *SagaContext) GetString(key string) string {
    if value, exists := ctx.Get(key); exists {
        if str, ok := value.(string); ok {
            return str
        }
    }
    return ""
}

func (ctx *SagaContext) GetDecimal(key string) decimal.Decimal {
    if value, exists := ctx.Get(key); exists {
        if dec, ok := value.(decimal.Decimal); ok {
            return dec
        }
    }
    return decimal.Zero
}

// Step interfaces
type SagaStep interface {
    GetStepID() string
    GetStepName() string
    Execute(ctx context.Context, sagaCtx *SagaContext) error
    CanCompensate() bool
    GetTimeout() time.Duration
}

type CompensatableStep interface {
    SagaStep
    Compensate(ctx context.Context, sagaCtx *SagaContext) error
}

// Base step implementation
type BaseSagaStep struct {
    StepID   string        `json:"step_id"`
    StepName string        `json:"step_name"`
    Timeout  time.Duration `json:"timeout"`
}

func (s *BaseSagaStep) GetStepID() string {
    return s.StepID
}

func (s *BaseSagaStep) GetStepName() string {
    return s.StepName
}

func (s *BaseSagaStep) GetTimeout() time.Duration {
    if s.Timeout == 0 {
        return time.Minute * 5 // default timeout
    }
    return s.Timeout
}

func (s *BaseSagaStep) CanCompensate() bool {
    return false // override in compensatable steps
}

// Saga execution record
type SagaExecution struct {
    SagaID       string                 `json:"saga_id"`
    SagaType     string                 `json:"saga_type"`
    State        SagaState              `json:"state"`
    CurrentStep  int                    `json:"current_step"`
    StepStates   map[string]StepState   `json:"step_states"`
    Context      *SagaContext           `json:"context"`
    StartedAt    time.Time              `json:"started_at"`
    CompletedAt  *time.Time             `json:"completed_at,omitempty"`
    FailedAt     *time.Time             `json:"failed_at,omitempty"`
    Error        *string                `json:"error,omitempty"`
    Compensations []CompensationRecord   `json:"compensations"`
}

type CompensationRecord struct {
    StepID       string    `json:"step_id"`
    CompensatedAt time.Time `json:"compensated_at"`
    Error        *string   `json:"error,omitempty"`
}

// Payment-specific steps
type ValidateUserStep struct {
    BaseSagaStep
    userService UserService
}

func NewValidateUserStep(userService UserService) *ValidateUserStep {
    return &ValidateUserStep{
        BaseSagaStep: BaseSagaStep{
            StepID:   "validate_user",
            StepName: "Validate User",
            Timeout:  time.Second * 30,
        },
        userService: userService,
    }
}

func (s *ValidateUserStep) Execute(ctx context.Context, sagaCtx *SagaContext) error {
    userID := sagaCtx.GetString("user_id")
    if userID == "" {
        return fmt.Errorf("user_id not found in saga context")
    }

    user, err := s.userService.ValidateUser(ctx, userID)
    if err != nil {
        return fmt.Errorf("user validation failed: %w", err)
    }

    sagaCtx.Set("user", user)
    sagaCtx.Set("user_validated", true)

    log.Printf("User validated: %s", userID)
    return nil
}

type ReserveFundsStep struct {
    BaseSagaStep
    accountService AccountService
}

func NewReserveFundsStep(accountService AccountService) *ReserveFundsStep {
    return &ReserveFundsStep{
        BaseSagaStep: BaseSagaStep{
            StepID:   "reserve_funds",
            StepName: "Reserve Funds",
            Timeout:  time.Minute * 2,
        },
        accountService: accountService,
    }
}

func (s *ReserveFundsStep) Execute(ctx context.Context, sagaCtx *SagaContext) error {
    userID := sagaCtx.GetString("user_id")
    amount := sagaCtx.GetDecimal("amount")

    if amount.LessThanOrEqual(decimal.Zero) {
        return fmt.Errorf("invalid amount: %v", amount)
    }

    reservationID, err := s.accountService.ReserveFunds(ctx, userID, amount)
    if err != nil {
        return fmt.Errorf("failed to reserve funds: %w", err)
    }

    sagaCtx.Set("reservation_id", reservationID)
    sagaCtx.Set("funds_reserved", true)

    log.Printf("Funds reserved: %s for amount %v", reservationID, amount)
    return nil
}

func (s *ReserveFundsStep) CanCompensate() bool {
    return true
}

func (s *ReserveFundsStep) Compensate(ctx context.Context, sagaCtx *SagaContext) error {
    reservationID := sagaCtx.GetString("reservation_id")
    if reservationID == "" {
        return nil // Nothing to compensate
    }

    err := s.accountService.ReleaseReservation(ctx, reservationID)
    if err != nil {
        return fmt.Errorf("failed to release reservation: %w", err)
    }

    sagaCtx.Set("funds_released", true)
    log.Printf("Funds reservation released: %s", reservationID)
    return nil
}

type ProcessPaymentStep struct {
    BaseSagaStep
    paymentService PaymentService
}

func NewProcessPaymentStep(paymentService PaymentService) *ProcessPaymentStep {
    return &ProcessPaymentStep{
        BaseSagaStep: BaseSagaStep{
            StepID:   "process_payment",
            StepName: "Process Payment",
            Timeout:  time.Minute * 5,
        },
        paymentService: paymentService,
    }
}

func (s *ProcessPaymentStep) Execute(ctx context.Context, sagaCtx *SagaContext) error {
    paymentID := sagaCtx.GetString("payment_id")
    userID := sagaCtx.GetString("user_id")
    merchantID := sagaCtx.GetString("merchant_id")
    amount := sagaCtx.GetDecimal("amount")
    reservationID := sagaCtx.GetString("reservation_id")

    transactionID, err := s.paymentService.ProcessPayment(ctx, ProcessPaymentRequest{
        PaymentID:     paymentID,
        UserID:        userID,
        MerchantID:    merchantID,
        Amount:        amount,
        ReservationID: reservationID,
    })

    if err != nil {
        return fmt.Errorf("payment processing failed: %w", err)
    }

    sagaCtx.Set("transaction_id", transactionID)
    sagaCtx.Set("payment_processed", true)

    log.Printf("Payment processed: %s -> %s", paymentID, transactionID)
    return nil
}

func (s *ProcessPaymentStep) CanCompensate() bool {
    return true
}

func (s *ProcessPaymentStep) Compensate(ctx context.Context, sagaCtx *SagaContext) error {
    transactionID := sagaCtx.GetString("transaction_id")
    if transactionID == "" {
        return nil // Nothing to compensate
    }

    refundID, err := s.paymentService.RefundPayment(ctx, transactionID)
    if err != nil {
        return fmt.Errorf("failed to refund payment: %w", err)
    }

    sagaCtx.Set("refund_id", refundID)
    sagaCtx.Set("payment_refunded", true)
    log.Printf("Payment refunded: %s -> %s", transactionID, refundID)
    return nil
}

type UpdateMerchantBalanceStep struct {
    BaseSagaStep
    merchantService MerchantService
}

func NewUpdateMerchantBalanceStep(merchantService MerchantService) *UpdateMerchantBalanceStep {
    return &UpdateMerchantBalanceStep{
        BaseSagaStep: BaseSagaStep{
            StepID:   "update_merchant_balance",
            StepName: "Update Merchant Balance",
            Timeout:  time.Minute * 2,
        },
        merchantService: merchantService,
    }
}

func (s *UpdateMerchantBalanceStep) Execute(ctx context.Context, sagaCtx *SagaContext) error {
    merchantID := sagaCtx.GetString("merchant_id")
    amount := sagaCtx.GetDecimal("amount")
    transactionID := sagaCtx.GetString("transaction_id")

    err := s.merchantService.CreditBalance(ctx, merchantID, amount, transactionID)
    if err != nil {
        return fmt.Errorf("failed to update merchant balance: %w", err)
    }

    sagaCtx.Set("merchant_balance_updated", true)
    log.Printf("Merchant balance updated: %s +%v", merchantID, amount)
    return nil
}

func (s *UpdateMerchantBalanceStep) CanCompensate() bool {
    return true
}

func (s *UpdateMerchantBalanceStep) Compensate(ctx context.Context, sagaCtx *SagaContext) error {
    merchantID := sagaCtx.GetString("merchant_id")
    amount := sagaCtx.GetDecimal("amount")
    transactionID := sagaCtx.GetString("transaction_id")

    err := s.merchantService.DebitBalance(ctx, merchantID, amount, transactionID+"_compensation")
    if err != nil {
        return fmt.Errorf("failed to compensate merchant balance: %w", err)
    }

    sagaCtx.Set("merchant_balance_compensated", true)
    log.Printf("Merchant balance compensated: %s -%v", merchantID, amount)
    return nil
}

type SendNotificationStep struct {
    BaseSagaStep
    notificationService NotificationService
}

func NewSendNotificationStep(notificationService NotificationService) *SendNotificationStep {
    return &SendNotificationStep{
        BaseSagaStep: BaseSagaStep{
            StepID:   "send_notification",
            StepName: "Send Notification",
            Timeout:  time.Minute,
        },
        notificationService: notificationService,
    }
}

func (s *SendNotificationStep) Execute(ctx context.Context, sagaCtx *SagaContext) error {
    userID := sagaCtx.GetString("user_id")
    merchantID := sagaCtx.GetString("merchant_id")
    amount := sagaCtx.GetDecimal("amount")
    transactionID := sagaCtx.GetString("transaction_id")

    // Send notification to user
    err := s.notificationService.SendPaymentSuccessNotification(ctx, userID, NotificationData{
        Type:          "payment_success",
        Amount:        amount,
        MerchantID:    merchantID,
        TransactionID: transactionID,
    })

    if err != nil {
        // Notification failure is not critical, log but don't fail the saga
        log.Printf("Failed to send user notification: %v", err)
    }

    // Send notification to merchant
    err = s.notificationService.SendPaymentReceivedNotification(ctx, merchantID, NotificationData{
        Type:          "payment_received",
        Amount:        amount,
        UserID:        userID,
        TransactionID: transactionID,
    })

    if err != nil {
        log.Printf("Failed to send merchant notification: %v", err)
    }

    sagaCtx.Set("notifications_sent", true)
    log.Printf("Notifications sent for transaction: %s", transactionID)
    return nil
}

// Saga definition
type PaymentSaga struct {
    SagaID    string          `json:"saga_id"`
    SagaType  string          `json:"saga_type"`
    Steps     []SagaStep      `json:"steps"`
    Context   *SagaContext    `json:"context"`
    Execution *SagaExecution  `json:"execution"`
}

func NewPaymentSaga(paymentID, userID, merchantID string, amount decimal.Decimal, services Services) *PaymentSaga {
    sagaID := uuid.New().String()
    sagaCtx := NewSagaContext(sagaID)

    // Initialize saga context
    sagaCtx.Set("payment_id", paymentID)
    sagaCtx.Set("user_id", userID)
    sagaCtx.Set("merchant_id", merchantID)
    sagaCtx.Set("amount", amount)

    steps := []SagaStep{
        NewValidateUserStep(services.UserService),
        NewReserveFundsStep(services.AccountService),
        NewProcessPaymentStep(services.PaymentService),
        NewUpdateMerchantBalanceStep(services.MerchantService),
        NewSendNotificationStep(services.NotificationService),
    }

    execution := &SagaExecution{
        SagaID:      sagaID,
        SagaType:    "payment",
        State:       SagaStateStarted,
        CurrentStep: 0,
        StepStates:  make(map[string]StepState),
        Context:     sagaCtx,
        StartedAt:   time.Now(),
        Compensations: make([]CompensationRecord, 0),
    }

    // Initialize step states
    for _, step := range steps {
        execution.StepStates[step.GetStepID()] = StepStatePending
    }

    return &PaymentSaga{
        SagaID:    sagaID,
        SagaType:  "payment",
        Steps:     steps,
        Context:   sagaCtx,
        Execution: execution,
    }
}

// Saga Orchestrator
type SagaOrchestrator struct {
    sagaStore    SagaStore
    eventBus     EventBus
    retryPolicy  RetryPolicy
    mu           sync.RWMutex
    runningSagas map[string]*PaymentSaga
}

func NewSagaOrchestrator(sagaStore SagaStore, eventBus EventBus) *SagaOrchestrator {
    return &SagaOrchestrator{
        sagaStore:    sagaStore,
        eventBus:     eventBus,
        retryPolicy:  NewRetryPolicy(3, time.Second*2, 2.0),
        runningSagas: make(map[string]*PaymentSaga),
    }
}

func (o *SagaOrchestrator) ExecuteSaga(ctx context.Context, saga *PaymentSaga) error {
    o.mu.Lock()
    o.runningSagas[saga.SagaID] = saga
    o.mu.Unlock()

    defer func() {
        o.mu.Lock()
        delete(o.runningSagas, saga.SagaID)
        o.mu.Unlock()
    }()

    saga.Execution.State = SagaStateInProgress
    if err := o.sagaStore.SaveSaga(ctx, saga); err != nil {
        return fmt.Errorf("failed to save saga: %w", err)
    }

    // Execute steps sequentially
    for i, step := range saga.Steps {
        saga.Execution.CurrentStep = i
        saga.Execution.StepStates[step.GetStepID()] = StepStateExecuting

        if err := o.sagaStore.SaveSaga(ctx, saga); err != nil {
            log.Printf("Failed to save saga state: %v", err)
        }

        // Execute step with timeout
        stepCtx, cancel := context.WithTimeout(ctx, step.GetTimeout())
        err := o.executeStepWithRetry(stepCtx, step, saga.Context)
        cancel()

        if err != nil {
            saga.Execution.StepStates[step.GetStepID()] = StepStateFailed
            saga.Execution.State = SagaStateFailed
            now := time.Now()
            saga.Execution.FailedAt = &now
            errorStr := err.Error()
            saga.Execution.Error = &errorStr

            if err := o.sagaStore.SaveSaga(ctx, saga); err != nil {
                log.Printf("Failed to save failed saga: %v", err)
            }

            // Start compensation
            return o.compensateSaga(ctx, saga, i-1)
        }

        saga.Execution.StepStates[step.GetStepID()] = StepStateCompleted

        // Publish step completed event
        o.publishStepCompletedEvent(saga.SagaID, step.GetStepID())
    }

    // All steps completed successfully
    saga.Execution.State = SagaStateCompleted
    now := time.Now()
    saga.Execution.CompletedAt = &now

    if err := o.sagaStore.SaveSaga(ctx, saga); err != nil {
        log.Printf("Failed to save completed saga: %v", err)
    }

    // Publish saga completed event
    o.publishSagaCompletedEvent(saga.SagaID)

    log.Printf("Saga completed successfully: %s", saga.SagaID)
    return nil
}

func (o *SagaOrchestrator) executeStepWithRetry(ctx context.Context, step SagaStep, sagaCtx *SagaContext) error {
    return o.retryPolicy.Execute(func() error {
        return step.Execute(ctx, sagaCtx)
    })
}

func (o *SagaOrchestrator) compensateSaga(ctx context.Context, saga *PaymentSaga, lastCompletedStep int) error {
    saga.Execution.State = SagaStateCompensating
    if err := o.sagaStore.SaveSaga(ctx, saga); err != nil {
        log.Printf("Failed to save compensating saga: %v", err)
    }

    log.Printf("Starting compensation for saga: %s from step %d", saga.SagaID, lastCompletedStep)

    // Compensate in reverse order
    for i := lastCompletedStep; i >= 0; i-- {
        step := saga.Steps[i]

        if compensatableStep, ok := step.(CompensatableStep); ok && step.CanCompensate() {
            stepID := step.GetStepID()
            saga.Execution.StepStates[stepID] = StepStateCompensating

            if err := o.sagaStore.SaveSaga(ctx, saga); err != nil {
                log.Printf("Failed to save saga state during compensation: %v", err)
            }

            compensationCtx, cancel := context.WithTimeout(ctx, step.GetTimeout())
            err := o.retryPolicy.Execute(func() error {
                return compensatableStep.Compensate(compensationCtx, saga.Context)
            })
            cancel()

            compensation := CompensationRecord{
                StepID:        stepID,
                CompensatedAt: time.Now(),
            }

            if err != nil {
                errorStr := err.Error()
                compensation.Error = &errorStr
                saga.Execution.StepStates[stepID] = StepStateFailed
                log.Printf("Compensation failed for step %s: %v", stepID, err)
            } else {
                saga.Execution.StepStates[stepID] = StepStateCompensated
                log.Printf("Compensation completed for step: %s", stepID)
            }

            saga.Execution.Compensations = append(saga.Execution.Compensations, compensation)
        }
    }

    saga.Execution.State = SagaStateCompensated
    if err := o.sagaStore.SaveSaga(ctx, saga); err != nil {
        log.Printf("Failed to save compensated saga: %v", err)
    }

    // Publish saga compensated event
    o.publishSagaCompensatedEvent(saga.SagaID)

    log.Printf("Saga compensation completed: %s", saga.SagaID)
    return fmt.Errorf("saga failed and was compensated")
}

func (o *SagaOrchestrator) publishStepCompletedEvent(sagaID, stepID string) {
    event := SagaStepCompletedEvent{
        SagaID:    sagaID,
        StepID:    stepID,
        Timestamp: time.Now(),
    }

    if err := o.eventBus.Publish(context.Background(), "saga.step.completed", event); err != nil {
        log.Printf("Failed to publish step completed event: %v", err)
    }
}

func (o *SagaOrchestrator) publishSagaCompletedEvent(sagaID string) {
    event := SagaCompletedEvent{
        SagaID:    sagaID,
        Timestamp: time.Now(),
    }

    if err := o.eventBus.Publish(context.Background(), "saga.completed", event); err != nil {
        log.Printf("Failed to publish saga completed event: %v", err)
    }
}

func (o *SagaOrchestrator) publishSagaCompensatedEvent(sagaID string) {
    event := SagaCompensatedEvent{
        SagaID:    sagaID,
        Timestamp: time.Now(),
    }

    if err := o.eventBus.Publish(context.Background(), "saga.compensated", event); err != nil {
        log.Printf("Failed to publish saga compensated event: %v", err)
    }
}

// Events
type SagaStepCompletedEvent struct {
    SagaID    string    `json:"saga_id"`
    StepID    string    `json:"step_id"`
    Timestamp time.Time `json:"timestamp"`
}

type SagaCompletedEvent struct {
    SagaID    string    `json:"saga_id"`
    Timestamp time.Time `json:"timestamp"`
}

type SagaCompensatedEvent struct {
    SagaID    string    `json:"saga_id"`
    Timestamp time.Time `json:"timestamp"`
}

// Service interfaces (external dependencies)
type UserService interface {
    ValidateUser(ctx context.Context, userID string) (*User, error)
}

type AccountService interface {
    ReserveFunds(ctx context.Context, userID string, amount decimal.Decimal) (string, error)
    ReleaseReservation(ctx context.Context, reservationID string) error
}

type PaymentService interface {
    ProcessPayment(ctx context.Context, req ProcessPaymentRequest) (string, error)
    RefundPayment(ctx context.Context, transactionID string) (string, error)
}

type MerchantService interface {
    CreditBalance(ctx context.Context, merchantID string, amount decimal.Decimal, transactionID string) error
    DebitBalance(ctx context.Context, merchantID string, amount decimal.Decimal, transactionID string) error
}

type NotificationService interface {
    SendPaymentSuccessNotification(ctx context.Context, userID string, data NotificationData) error
    SendPaymentReceivedNotification(ctx context.Context, merchantID string, data NotificationData) error
}

type SagaStore interface {
    SaveSaga(ctx context.Context, saga *PaymentSaga) error
    LoadSaga(ctx context.Context, sagaID string) (*PaymentSaga, error)
    DeleteSaga(ctx context.Context, sagaID string) error
}

type EventBus interface {
    Publish(ctx context.Context, topic string, event interface{}) error
}

// Data structures
type User struct {
    UserID   string `json:"user_id"`
    Name     string `json:"name"`
    Email    string `json:"email"`
    Active   bool   `json:"active"`
}

type ProcessPaymentRequest struct {
    PaymentID     string          `json:"payment_id"`
    UserID        string          `json:"user_id"`
    MerchantID    string          `json:"merchant_id"`
    Amount        decimal.Decimal `json:"amount"`
    ReservationID string          `json:"reservation_id"`
}

type NotificationData struct {
    Type          string          `json:"type"`
    Amount        decimal.Decimal `json:"amount"`
    UserID        string          `json:"user_id,omitempty"`
    MerchantID    string          `json:"merchant_id,omitempty"`
    TransactionID string          `json:"transaction_id"`
}

type Services struct {
    UserService         UserService
    AccountService      AccountService
    PaymentService      PaymentService
    MerchantService     MerchantService
    NotificationService NotificationService
}

// Retry Policy
type RetryPolicy struct {
    MaxAttempts int
    InitialDelay time.Duration
    Multiplier   float64
}

func NewRetryPolicy(maxAttempts int, initialDelay time.Duration, multiplier float64) RetryPolicy {
    return RetryPolicy{
        MaxAttempts:  maxAttempts,
        InitialDelay: initialDelay,
        Multiplier:   multiplier,
    }
}

func (r RetryPolicy) Execute(fn func() error) error {
    var lastErr error
    delay := r.InitialDelay

    for attempt := 1; attempt <= r.MaxAttempts; attempt++ {
        if err := fn(); err != nil {
            lastErr = err
            if attempt == r.MaxAttempts {
                break
            }

            log.Printf("Attempt %d failed: %v. Retrying in %v", attempt, err, delay)
            time.Sleep(delay)
            delay = time.Duration(float64(delay) * r.Multiplier)
        } else {
            return nil // Success
        }
    }

    return fmt.Errorf("failed after %d attempts: %w", r.MaxAttempts, lastErr)
}

// Mock implementations for demo
type MockUserService struct{}

func (m *MockUserService) ValidateUser(ctx context.Context, userID string) (*User, error) {
    if userID == "invalid_user" {
        return nil, fmt.Errorf("user not found")
    }

    return &User{
        UserID: userID,
        Name:   "John Doe",
        Email:  "john@example.com",
        Active: true,
    }, nil
}

type MockAccountService struct {
    reservations map[string]decimal.Decimal
    mu           sync.Mutex
}

func NewMockAccountService() *MockAccountService {
    return &MockAccountService{
        reservations: make(map[string]decimal.Decimal),
    }
}

func (m *MockAccountService) ReserveFunds(ctx context.Context, userID string, amount decimal.Decimal) (string, error) {
    if amount.GreaterThan(decimal.NewFromFloat(1000)) {
        return "", fmt.Errorf("insufficient funds")
    }

    reservationID := uuid.New().String()

    m.mu.Lock()
    m.reservations[reservationID] = amount
    m.mu.Unlock()

    return reservationID, nil
}

func (m *MockAccountService) ReleaseReservation(ctx context.Context, reservationID string) error {
    m.mu.Lock()
    defer m.mu.Unlock()

    if _, exists := m.reservations[reservationID]; !exists {
        return fmt.Errorf("reservation not found: %s", reservationID)
    }

    delete(m.reservations, reservationID)
    return nil
}

type MockPaymentService struct{}

func (m *MockPaymentService) ProcessPayment(ctx context.Context, req ProcessPaymentRequest) (string, error) {
    if req.Amount.GreaterThan(decimal.NewFromFloat(500)) {
        return "", fmt.Errorf("payment amount too high")
    }

    return uuid.New().String(), nil
}

func (m *MockPaymentService) RefundPayment(ctx context.Context, transactionID string) (string, error) {
    return uuid.New().String(), nil
}

type MockMerchantService struct{}

func (m *MockMerchantService) CreditBalance(ctx context.Context, merchantID string, amount decimal.Decimal, transactionID string) error {
    return nil
}

func (m *MockMerchantService) DebitBalance(ctx context.Context, merchantID string, amount decimal.Decimal, transactionID string) error {
    return nil
}

type MockNotificationService struct{}

func (m *MockNotificationService) SendPaymentSuccessNotification(ctx context.Context, userID string, data NotificationData) error {
    return nil
}

func (m *MockNotificationService) SendPaymentReceivedNotification(ctx context.Context, merchantID string, data NotificationData) error {
    return nil
}

type InMemorySagaStore struct {
    sagas map[string]*PaymentSaga
    mu    sync.RWMutex
}

func NewInMemorySagaStore() *InMemorySagaStore {
    return &InMemorySagaStore{
        sagas: make(map[string]*PaymentSaga),
    }
}

func (s *InMemorySagaStore) SaveSaga(ctx context.Context, saga *PaymentSaga) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    // Deep copy to avoid race conditions
    sagaCopy := *saga
    s.sagas[saga.SagaID] = &sagaCopy
    return nil
}

func (s *InMemorySagaStore) LoadSaga(ctx context.Context, sagaID string) (*PaymentSaga, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()

    saga, exists := s.sagas[sagaID]
    if !exists {
        return nil, fmt.Errorf("saga not found: %s", sagaID)
    }

    // Return a copy
    sagaCopy := *saga
    return &sagaCopy, nil
}

func (s *InMemorySagaStore) DeleteSaga(ctx context.Context, sagaID string) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    delete(s.sagas, sagaID)
    return nil
}

type MockEventBus struct{}

func (m *MockEventBus) Publish(ctx context.Context, topic string, event interface{}) error {
    log.Printf("Event published to topic %s: %+v", topic, event)
    return nil
}

// Example usage
func main() {
    fmt.Println("=== Saga Pattern Demo ===\n")

    // Setup services
    services := Services{
        UserService:         &MockUserService{},
        AccountService:      NewMockAccountService(),
        PaymentService:      &MockPaymentService{},
        MerchantService:     &MockMerchantService{},
        NotificationService: &MockNotificationService{},
    }

    // Setup infrastructure
    sagaStore := NewInMemorySagaStore()
    eventBus := &MockEventBus{}
    orchestrator := NewSagaOrchestrator(sagaStore, eventBus)

    ctx := context.Background()

    // Example 1: Successful payment saga
    fmt.Println("1. Successful Payment Saga")
    paymentSaga1 := NewPaymentSaga(
        "PAY_001",
        "USER_123",
        "MERCHANT_456",
        decimal.NewFromFloat(100.50),
        services,
    )

    err := orchestrator.ExecuteSaga(ctx, paymentSaga1)
    if err != nil {
        fmt.Printf("Saga failed: %v\n", err)
    } else {
        fmt.Printf("Saga completed successfully: %s\n", paymentSaga1.SagaID)
    }

    fmt.Printf("Final state: %s\n", paymentSaga1.Execution.State)
    fmt.Printf("Steps completed: %d/%d\n", paymentSaga1.Execution.CurrentStep+1, len(paymentSaga1.Steps))

    fmt.Println()

    // Example 2: Failed payment saga (amount too high)
    fmt.Println("2. Failed Payment Saga (Amount Too High)")
    paymentSaga2 := NewPaymentSaga(
        "PAY_002",
        "USER_123",
        "MERCHANT_456",
        decimal.NewFromFloat(600.00), // This will fail in ProcessPaymentStep
        services,
    )

    err = orchestrator.ExecuteSaga(ctx, paymentSaga2)
    if err != nil {
        fmt.Printf("Saga failed as expected: %v\n", err)
    }

    fmt.Printf("Final state: %s\n", paymentSaga2.Execution.State)
    fmt.Printf("Failed at step: %d\n", paymentSaga2.Execution.CurrentStep)
    fmt.Printf("Compensations: %d\n", len(paymentSaga2.Execution.Compensations))

    if paymentSaga2.Execution.Error != nil {
        fmt.Printf("Error: %s\n", *paymentSaga2.Execution.Error)
    }

    fmt.Println()

    // Example 3: Failed payment saga (insufficient funds)
    fmt.Println("3. Failed Payment Saga (Insufficient Funds)")
    paymentSaga3 := NewPaymentSaga(
        "PAY_003",
        "USER_123",
        "MERCHANT_456",
        decimal.NewFromFloat(1500.00), // This will fail in ReserveFundsStep
        services,
    )

    err = orchestrator.ExecuteSaga(ctx, paymentSaga3)
    if err != nil {
        fmt.Printf("Saga failed as expected: %v\n", err)
    }

    fmt.Printf("Final state: %s\n", paymentSaga3.Execution.State)
    fmt.Printf("Failed at step: %d\n", paymentSaga3.Execution.CurrentStep)

    // Check step states
    fmt.Println("Step states:")
    for _, step := range paymentSaga3.Steps {
        state := paymentSaga3.Execution.StepStates[step.GetStepID()]
        fmt.Printf("  %s: %s\n", step.GetStepName(), state)
    }

    fmt.Println()

    // Example 4: Show saga persistence
    fmt.Println("4. Saga Persistence")
    savedSaga, err := sagaStore.LoadSaga(ctx, paymentSaga1.SagaID)
    if err != nil {
        fmt.Printf("Failed to load saga: %v\n", err)
    } else {
        fmt.Printf("Loaded saga: %s\n", savedSaga.SagaID)
        fmt.Printf("State: %s\n", savedSaga.Execution.State)
        fmt.Printf("Completed at: %v\n", savedSaga.Execution.CompletedAt)
    }

    fmt.Println("\n=== Saga Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Orchestrator-based Saga (Centralized)**

```go
type SagaOrchestrator struct {
    steps    []SagaStep
    context  *SagaContext
    state    SagaState
}

// Central coordinator controls all steps
func (o *SagaOrchestrator) Execute() error {
    for _, step := range o.steps {
        if err := step.Execute(o.context); err != nil {
            return o.compensate()
        }
    }
    return nil
}
```

2. **Choreography-based Saga (Distributed)**

```go
type SagaStep struct {
    OnSuccess func() Event
    OnFailure func() Event
}

// Each service knows what to do next
func (s *PaymentService) ProcessPayment(event PaymentEvent) {
    if err := s.process(event); err != nil {
        s.eventBus.Publish(PaymentFailedEvent{})
    } else {
        s.eventBus.Publish(PaymentSuccessEvent{})
    }
}
```

3. **State Machine Saga**

```go
type SagaStateMachine struct {
    states      map[SagaState]StateHandler
    transitions map[SagaState]map[Event]SagaState
    current     SagaState
}

func (sm *SagaStateMachine) HandleEvent(event Event) error {
    nextState, exists := sm.transitions[sm.current][event]
    if !exists {
        return fmt.Errorf("invalid transition")
    }

    handler := sm.states[nextState]
    if err := handler.Handle(event); err != nil {
        return err
    }

    sm.current = nextState
    return nil
}
```

4. **Event-Sourced Saga**

```go
type EventSourcedSaga struct {
    sagaID string
    events []SagaEvent
    state  SagaState
}

func (s *EventSourcedSaga) ApplyEvent(event SagaEvent) {
    s.events = append(s.events, event)
    s.state = s.computeState(s.events)
}

func (s *EventSourcedSaga) computeState(events []SagaEvent) SagaState {
    // Replay events to compute current state
    state := SagaStateStarted
    for _, event := range events {
        state = s.transition(state, event)
    }
    return state
}
```

### Trade-offs

**Pros:**

- **Distributed Consistency**: Maintains consistency across services
- **Fault Tolerance**: Automatic compensation on failures
- **Long-Running Support**: Handles long-running processes
- **Resilience**: No distributed locks or two-phase commits
- **Observability**: Clear audit trail of operations

**Cons:**

- **Complexity**: More complex than local transactions
- **Eventual Consistency**: Not immediately consistent
- **Compensation Logic**: Must implement compensation for each step
- **Debugging**: Harder to debug distributed flows
- **Programming Model**: Different from traditional transactions

**When to Choose Saga vs Alternatives:**

| Scenario                    | Pattern              | Reason                              |
| --------------------------- | -------------------- | ----------------------------------- |
| Distributed transactions    | Saga                 | Handles distributed consistency     |
| Local transactions          | Database Transaction | Simpler and ACID compliant          |
| Eventual consistency OK     | Saga                 | Better performance and availability |
| Strong consistency required | Two-Phase Commit     | Stronger consistency guarantees     |
| Long-running processes      | Saga                 | Doesn't hold locks                  |
| Short operations            | Local Transaction    | Lower overhead                      |

## Testable Example

```go
package main

import (
    "context"
    "testing"
    "time"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/stretchr/testify/mock"
    "github.com/shopspring/decimal"
)

// Mock services for testing
type MockUserServiceTest struct {
    mock.Mock
}

func (m *MockUserServiceTest) ValidateUser(ctx context.Context, userID string) (*User, error) {
    args := m.Called(ctx, userID)
    return args.Get(0).(*User), args.Error(1)
}

type MockAccountServiceTest struct {
    mock.Mock
}

func (m *MockAccountServiceTest) ReserveFunds(ctx context.Context, userID string, amount decimal.Decimal) (string, error) {
    args := m.Called(ctx, userID, amount)
    return args.String(0), args.Error(1)
}

func (m *MockAccountServiceTest) ReleaseReservation(ctx context.Context, reservationID string) error {
    args := m.Called(ctx, reservationID)
    return args.Error(0)
}

type MockPaymentServiceTest struct {
    mock.Mock
}

func (m *MockPaymentServiceTest) ProcessPayment(ctx context.Context, req ProcessPaymentRequest) (string, error) {
    args := m.Called(ctx, req)
    return args.String(0), args.Error(1)
}

func (m *MockPaymentServiceTest) RefundPayment(ctx context.Context, transactionID string) (string, error) {
    args := m.Called(ctx, transactionID)
    return args.String(0), args.Error(1)
}

func TestPaymentSaga_SuccessfulExecution(t *testing.T) {
    // Setup mocks
    userService := &MockUserServiceTest{}
    accountService := &MockAccountServiceTest{}
    paymentService := &MockPaymentServiceTest{}
    merchantService := &MockMerchantService{}
    notificationService := &MockNotificationService{}

    services := Services{
        UserService:         userService,
        AccountService:      accountService,
        PaymentService:      paymentService,
        MerchantService:     merchantService,
        NotificationService: notificationService,
    }

    // Setup expectations
    user := &User{UserID: "USER_123", Name: "John Doe", Active: true}
    userService.On("ValidateUser", mock.Anything, "USER_123").Return(user, nil)

    accountService.On("ReserveFunds", mock.Anything, "USER_123", decimal.NewFromFloat(100.0)).
        Return("RESERVATION_123", nil)

    paymentService.On("ProcessPayment", mock.Anything, mock.AnythingOfType("ProcessPaymentRequest")).
        Return("TRANSACTION_123", nil)

    // Create and execute saga
    saga := NewPaymentSaga("PAY_001", "USER_123", "MERCHANT_456", decimal.NewFromFloat(100.0), services)

    sagaStore := NewInMemorySagaStore()
    eventBus := &MockEventBus{}
    orchestrator := NewSagaOrchestrator(sagaStore, eventBus)

    ctx := context.Background()
    err := orchestrator.ExecuteSaga(ctx, saga)

    // Assertions
    assert.NoError(t, err)
    assert.Equal(t, SagaStateCompleted, saga.Execution.State)
    assert.NotNil(t, saga.Execution.CompletedAt)
    assert.Nil(t, saga.Execution.FailedAt)
    assert.Nil(t, saga.Execution.Error)

    // Verify all steps completed
    for _, step := range saga.Steps {
        state := saga.Execution.StepStates[step.GetStepID()]
        assert.Equal(t, StepStateCompleted, state)
    }

    // Verify context data
    assert.Equal(t, "RESERVATION_123", saga.Context.GetString("reservation_id"))
    assert.Equal(t, "TRANSACTION_123", saga.Context.GetString("transaction_id"))
    assert.True(t, saga.Context.Get("user_validated") != nil)

    // Verify mock expectations
    userService.AssertExpectations(t)
    accountService.AssertExpectations(t)
    paymentService.AssertExpectations(t)
}

func TestPaymentSaga_FailureAndCompensation(t *testing.T) {
    // Setup mocks
    userService := &MockUserServiceTest{}
    accountService := &MockAccountServiceTest{}
    paymentService := &MockPaymentServiceTest{}
    merchantService := &MockMerchantService{}
    notificationService := &MockNotificationService{}

    services := Services{
        UserService:         userService,
        AccountService:      accountService,
        PaymentService:      paymentService,
        MerchantService:     merchantService,
        NotificationService: notificationService,
    }

    // Setup expectations - payment will fail
    user := &User{UserID: "USER_123", Name: "John Doe", Active: true}
    userService.On("ValidateUser", mock.Anything, "USER_123").Return(user, nil)

    accountService.On("ReserveFunds", mock.Anything, "USER_123", decimal.NewFromFloat(100.0)).
        Return("RESERVATION_123", nil)

    paymentService.On("ProcessPayment", mock.Anything, mock.AnythingOfType("ProcessPaymentRequest")).
        Return("", fmt.Errorf("payment failed"))

    // Compensation expectations
    accountService.On("ReleaseReservation", mock.Anything, "RESERVATION_123").Return(nil)

    // Create and execute saga
    saga := NewPaymentSaga("PAY_001", "USER_123", "MERCHANT_456", decimal.NewFromFloat(100.0), services)

    sagaStore := NewInMemorySagaStore()
    eventBus := &MockEventBus{}
    orchestrator := NewSagaOrchestrator(sagaStore, eventBus)

    ctx := context.Background()
    err := orchestrator.ExecuteSaga(ctx, saga)

    // Assertions
    assert.Error(t, err)
    assert.Equal(t, SagaStateCompensated, saga.Execution.State)
    assert.Nil(t, saga.Execution.CompletedAt)
    assert.NotNil(t, saga.Execution.FailedAt)
    assert.NotNil(t, saga.Execution.Error)
    assert.Contains(t, *saga.Execution.Error, "payment failed")

    // Verify step states
    assert.Equal(t, StepStateCompleted, saga.Execution.StepStates["validate_user"])
    assert.Equal(t, StepStateCompensated, saga.Execution.StepStates["reserve_funds"])
    assert.Equal(t, StepStateFailed, saga.Execution.StepStates["process_payment"])

    // Verify compensations
    assert.Len(t, saga.Execution.Compensations, 1)
    assert.Equal(t, "reserve_funds", saga.Execution.Compensations[0].StepID)
    assert.Nil(t, saga.Execution.Compensations[0].Error)

    // Verify mock expectations
    userService.AssertExpectations(t)
    accountService.AssertExpectations(t)
    paymentService.AssertExpectations(t)
}

func TestPaymentSaga_EarlyFailure(t *testing.T) {
    // Setup mocks
    userService := &MockUserServiceTest{}
    accountService := &MockAccountServiceTest{}
    paymentService := &MockPaymentServiceTest{}
    merchantService := &MockMerchantService{}
    notificationService := &MockNotificationService{}

    services := Services{
        UserService:         userService,
        AccountService:      accountService,
        PaymentService:      paymentService,
        MerchantService:     merchantService,
        NotificationService: notificationService,
    }

    // Setup expectations - user validation will fail
    userService.On("ValidateUser", mock.Anything, "INVALID_USER").
        Return((*User)(nil), fmt.Errorf("user not found"))

    // Create and execute saga
    saga := NewPaymentSaga("PAY_001", "INVALID_USER", "MERCHANT_456", decimal.NewFromFloat(100.0), services)

    sagaStore := NewInMemorySagaStore()
    eventBus := &MockEventBus{}
    orchestrator := NewSagaOrchestrator(sagaStore, eventBus)

    ctx := context.Background()
    err := orchestrator.ExecuteSaga(ctx, saga)

    // Assertions
    assert.Error(t, err)
    assert.Equal(t, SagaStateCompensated, saga.Execution.State)
    assert.Equal(t, StepStateFailed, saga.Execution.StepStates["validate_user"])

    // No compensations should occur for first step failure
    assert.Len(t, saga.Execution.Compensations, 0)

    // Verify mock expectations
    userService.AssertExpectations(t)

    // Verify other services were not called
    accountService.AssertNotCalled(t, "ReserveFunds")
    paymentService.AssertNotCalled(t, "ProcessPayment")
}

func TestSagaContext_DataManagement(t *testing.T) {
    sagaCtx := NewSagaContext("SAGA_123")

    // Test setting and getting different types
    sagaCtx.Set("string_value", "test")
    sagaCtx.Set("decimal_value", decimal.NewFromFloat(100.50))
    sagaCtx.Set("bool_value", true)

    // Test string retrieval
    assert.Equal(t, "test", sagaCtx.GetString("string_value"))
    assert.Equal(t, "", sagaCtx.GetString("nonexistent"))

    // Test decimal retrieval
    assert.Equal(t, decimal.NewFromFloat(100.50), sagaCtx.GetDecimal("decimal_value"))
    assert.Equal(t, decimal.Zero, sagaCtx.GetDecimal("nonexistent"))

    // Test generic retrieval
    value, exists := sagaCtx.Get("bool_value")
    assert.True(t, exists)
    assert.Equal(t, true, value)

    _, exists = sagaCtx.Get("nonexistent")
    assert.False(t, exists)

    // Test timestamp updates
    originalTime := sagaCtx.UpdatedAt
    time.Sleep(1 * time.Millisecond)
    sagaCtx.Set("new_value", "updated")
    assert.True(t, sagaCtx.UpdatedAt.After(originalTime))
}

func TestSagaStore_Persistence(t *testing.T) {
    store := NewInMemorySagaStore()
    ctx := context.Background()

    // Create test saga
    services := Services{} // Empty services for test
    saga := NewPaymentSaga("PAY_001", "USER_123", "MERCHANT_456", decimal.NewFromFloat(100.0), services)
    saga.Execution.State = SagaStateInProgress

    // Test save
    err := store.SaveSaga(ctx, saga)
    assert.NoError(t, err)

    // Test load
    loadedSaga, err := store.LoadSaga(ctx, saga.SagaID)
    assert.NoError(t, err)
    assert.Equal(t, saga.SagaID, loadedSaga.SagaID)
    assert.Equal(t, saga.Execution.State, loadedSaga.Execution.State)

    // Test load non-existent
    _, err = store.LoadSaga(ctx, "NON_EXISTENT")
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "saga not found")

    // Test delete
    err = store.DeleteSaga(ctx, saga.SagaID)
    assert.NoError(t, err)

    // Verify deletion
    _, err = store.LoadSaga(ctx, saga.SagaID)
    assert.Error(t, err)
}

func TestRetryPolicy_Execution(t *testing.T) {
    retryPolicy := NewRetryPolicy(3, time.Millisecond*10, 2.0)

    t.Run("successful execution", func(t *testing.T) {
        callCount := 0
        err := retryPolicy.Execute(func() error {
            callCount++
            return nil
        })

        assert.NoError(t, err)
        assert.Equal(t, 1, callCount)
    })

    t.Run("eventual success", func(t *testing.T) {
        callCount := 0
        err := retryPolicy.Execute(func() error {
            callCount++
            if callCount < 3 {
                return fmt.Errorf("temporary error")
            }
            return nil
        })

        assert.NoError(t, err)
        assert.Equal(t, 3, callCount)
    })

    t.Run("max retries exceeded", func(t *testing.T) {
        callCount := 0
        err := retryPolicy.Execute(func() error {
            callCount++
            return fmt.Errorf("persistent error")
        })

        assert.Error(t, err)
        assert.Equal(t, 3, callCount)
        assert.Contains(t, err.Error(), "failed after 3 attempts")
    })
}

func TestSagaStep_Execution(t *testing.T) {
    userService := &MockUserServiceTest{}
    step := NewValidateUserStep(userService)

    sagaCtx := NewSagaContext("SAGA_123")
    sagaCtx.Set("user_id", "USER_123")

    user := &User{UserID: "USER_123", Name: "John Doe", Active: true}
    userService.On("ValidateUser", mock.Anything, "USER_123").Return(user, nil)

    ctx := context.Background()
    err := step.Execute(ctx, sagaCtx)

    assert.NoError(t, err)
    assert.Equal(t, true, sagaCtx.Data["user_validated"])
    assert.Equal(t, user, sagaCtx.Data["user"])

    userService.AssertExpectations(t)
}

func TestCompensatableStep_Compensation(t *testing.T) {
    accountService := &MockAccountServiceTest{}
    step := NewReserveFundsStep(accountService)

    sagaCtx := NewSagaContext("SAGA_123")
    sagaCtx.Set("user_id", "USER_123")
    sagaCtx.Set("amount", decimal.NewFromFloat(100.0))

    // First execute the step
    accountService.On("ReserveFunds", mock.Anything, "USER_123", decimal.NewFromFloat(100.0)).
        Return("RESERVATION_123", nil)

    ctx := context.Background()
    err := step.Execute(ctx, sagaCtx)
    assert.NoError(t, err)
    assert.Equal(t, "RESERVATION_123", sagaCtx.GetString("reservation_id"))

    // Then compensate
    accountService.On("ReleaseReservation", mock.Anything, "RESERVATION_123").Return(nil)

    compensatableStep := step.(CompensatableStep)
    err = compensatableStep.Compensate(ctx, sagaCtx)
    assert.NoError(t, err)
    assert.Equal(t, true, sagaCtx.Data["funds_released"])

    accountService.AssertExpectations(t)
}

func BenchmarkSagaExecution(b *testing.B) {
    services := Services{
        UserService:         &MockUserService{},
        AccountService:      NewMockAccountService(),
        PaymentService:      &MockPaymentService{},
        MerchantService:     &MockMerchantService{},
        NotificationService: &MockNotificationService{},
    }

    sagaStore := NewInMemorySagaStore()
    eventBus := &MockEventBus{}
    orchestrator := NewSagaOrchestrator(sagaStore, eventBus)

    ctx := context.Background()

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        saga := NewPaymentSaga(
            fmt.Sprintf("PAY_%d", i),
            "USER_123",
            "MERCHANT_456",
            decimal.NewFromFloat(100.0),
            services,
        )

        err := orchestrator.ExecuteSaga(ctx, saga)
        if err != nil {
            b.Fatal(err)
        }
    }
}

func BenchmarkSagaContextDataAccess(b *testing.B) {
    sagaCtx := NewSagaContext("SAGA_123")

    // Setup test data
    for i := 0; i < 1000; i++ {
        sagaCtx.Set(fmt.Sprintf("key_%d", i), fmt.Sprintf("value_%d", i))
    }

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        key := fmt.Sprintf("key_%d", i%1000)
        value := sagaCtx.GetString(key)
        if value == "" {
            b.Fatal("Expected value not found")
        }
    }
}
```

## Integration Tips

### 1. Database Integration

```go
// PostgreSQL Saga Store
type PostgreSQLSagaStore struct {
    db *sql.DB
}

func (s *PostgreSQLSagaStore) SaveSaga(ctx context.Context, saga *PaymentSaga) error {
    sagaData, err := json.Marshal(saga)
    if err != nil {
        return err
    }

    _, err = s.db.ExecContext(ctx, `
        INSERT INTO sagas (saga_id, saga_type, state, current_step, saga_data, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (saga_id) DO UPDATE SET
            state = EXCLUDED.state,
            current_step = EXCLUDED.current_step,
            saga_data = EXCLUDED.saga_data,
            updated_at = EXCLUDED.updated_at
    `, saga.SagaID, saga.SagaType, saga.Execution.State, saga.Execution.CurrentStep,
       sagaData, saga.Context.CreatedAt, saga.Context.UpdatedAt)

    return err
}

func (s *PostgreSQLSagaStore) LoadSaga(ctx context.Context, sagaID string) (*PaymentSaga, error) {
    var sagaData []byte

    err := s.db.QueryRowContext(ctx, `
        SELECT saga_data FROM sagas WHERE saga_id = $1
    `, sagaID).Scan(&sagaData)

    if err != nil {
        return nil, err
    }

    var saga PaymentSaga
    if err := json.Unmarshal(sagaData, &saga); err != nil {
        return nil, err
    }

    return &saga, nil
}
```

### 2. Kafka Integration

```go
type KafkaSagaEventBus struct {
    producer *kafka.Writer
    consumer *kafka.Reader
}

func (k *KafkaSagaEventBus) Publish(ctx context.Context, topic string, event interface{}) error {
    eventData, err := json.Marshal(event)
    if err != nil {
        return err
    }

    message := kafka.Message{
        Topic: topic,
        Key:   []byte(fmt.Sprintf("saga_%s", getSagaID(event))),
        Value: eventData,
        Headers: []kafka.Header{
            {Key: "event_type", Value: []byte(getEventType(event))},
            {Key: "timestamp", Value: []byte(time.Now().Format(time.RFC3339))},
        },
    }

    return k.producer.WriteMessages(ctx, message)
}

func (k *KafkaSagaEventBus) StartConsumer(ctx context.Context, handler func(string, []byte) error) {
    for {
        message, err := k.consumer.ReadMessage(ctx)
        if err != nil {
            log.Printf("Error reading message: %v", err)
            continue
        }

        topic := message.Topic
        if err := handler(topic, message.Value); err != nil {
            log.Printf("Error handling message: %v", err)
        }
    }
}
```

### 3. Monitoring Integration

```go
type InstrumentedSagaOrchestrator struct {
    *SagaOrchestrator
    metrics SagaMetrics
}

type SagaMetrics interface {
    IncSagaStarted(sagaType string)
    IncSagaCompleted(sagaType string)
    IncSagaFailed(sagaType string)
    IncSagaCompensated(sagaType string)
    ObserveSagaDuration(sagaType string, duration time.Duration)
    IncStepExecuted(stepType string)
    IncStepFailed(stepType string)
    ObserveStepDuration(stepType string, duration time.Duration)
}

func (i *InstrumentedSagaOrchestrator) ExecuteSaga(ctx context.Context, saga *PaymentSaga) error {
    start := time.Now()
    i.metrics.IncSagaStarted(saga.SagaType)

    err := i.SagaOrchestrator.ExecuteSaga(ctx, saga)
    duration := time.Since(start)

    i.metrics.ObserveSagaDuration(saga.SagaType, duration)

    switch saga.Execution.State {
    case SagaStateCompleted:
        i.metrics.IncSagaCompleted(saga.SagaType)
    case SagaStateFailed:
        i.metrics.IncSagaFailed(saga.SagaType)
    case SagaStateCompensated:
        i.metrics.IncSagaCompensated(saga.SagaType)
    }

    return err
}
```

### 4. Timeout and Circuit Breaker Integration

```go
type ResilientSagaStep struct {
    SagaStep
    circuitBreaker *CircuitBreaker
    timeout        time.Duration
}

func (r *ResilientSagaStep) Execute(ctx context.Context, sagaCtx *SagaContext) error {
    // Add timeout
    timeoutCtx, cancel := context.WithTimeout(ctx, r.timeout)
    defer cancel()

    // Execute through circuit breaker
    return r.circuitBreaker.Execute(func() error {
        return r.SagaStep.Execute(timeoutCtx, sagaCtx)
    })
}

type SagaStepCircuitBreaker struct {
    failureThreshold int
    resetTimeout     time.Duration
    state           CircuitState
    failures        int
    lastFailureTime time.Time
}

func (cb *SagaStepCircuitBreaker) Execute(fn func() error) error {
    if cb.state == CircuitStateOpen {
        if time.Since(cb.lastFailureTime) > cb.resetTimeout {
            cb.state = CircuitStateHalfOpen
        } else {
            return fmt.Errorf("circuit breaker is open")
        }
    }

    err := fn()

    if err != nil {
        cb.failures++
        cb.lastFailureTime = time.Now()

        if cb.failures >= cb.failureThreshold {
            cb.state = CircuitStateOpen
        }

        return err
    }

    // Success - reset circuit breaker
    cb.failures = 0
    cb.state = CircuitStateClosed
    return nil
}
```

## Common Interview Questions

### 1. **What's the difference between Saga and Two-Phase Commit (2PC)?**

**Answer:**
| Aspect | Saga | Two-Phase Commit |
|--------|------|------------------|
| **Consistency** | Eventual consistency | Strong consistency |
| **Locking** | No distributed locks | Requires distributed locks |
| **Availability** | High availability | Lower availability (blocking) |
| **Failure Handling** | Compensation | Rollback |
| **Performance** | Better performance | Slower due to coordination |
| **Complexity** | Complex compensation logic | Complex coordination protocol |

**Saga Example:**

```go
// Non-blocking, compensatable
func (s *PaymentSaga) Execute() error {
    for _, step := range s.steps {
        if err := step.Execute(); err != nil {
            s.compensate() // Rollback via compensation
            return err
        }
    }
    return nil
}
```

**2PC Example:**

```go
// Blocking, atomic
func (t *TwoPhaseCommit) Execute() error {
    // Phase 1: Prepare
    for _, participant := range t.participants {
        if !participant.Prepare() {
            t.abort() // All participants rollback
            return errors.New("prepare failed")
        }
    }

    // Phase 2: Commit
    for _, participant := range t.participants {
        participant.Commit()
    }
    return nil
}
```

### 2. **How do you handle partial failures in Saga compensation?**

**Answer:**
Partial compensation failures require several strategies:

1. **Idempotent Compensation**: Make compensation operations idempotent

```go
func (s *ReserveFundsStep) Compensate(ctx context.Context, sagaCtx *SagaContext) error {
    reservationID := sagaCtx.GetString("reservation_id")

    // Idempotent - safe to call multiple times
    if reservationID == "" {
        return nil // Already compensated or nothing to compensate
    }

    return s.accountService.ReleaseReservation(ctx, reservationID)
}
```

2. **Compensation Retry**: Retry failed compensations

```go
func (o *SagaOrchestrator) compensateWithRetry(step CompensatableStep, sagaCtx *SagaContext) error {
    return o.retryPolicy.Execute(func() error {
        return step.Compensate(context.Background(), sagaCtx)
    })
}
```

3. **Manual Intervention**: Flag for manual resolution

```go
type CompensationRecord struct {
    StepID              string    `json:"step_id"`
    CompensatedAt       time.Time `json:"compensated_at"`
    Error               *string   `json:"error,omitempty"`
    RequiresIntervention bool      `json:"requires_intervention"`
}
```

4. **Dead Letter Queue**: Send failed compensations to DLQ

```go
if err := step.Compensate(ctx, sagaCtx); err != nil {
    compensationEvent := CompensationFailedEvent{
        SagaID: saga.SagaID,
        StepID: step.GetStepID(),
        Error:  err.Error(),
    }

    // Send to dead letter queue for manual processing
    o.deadLetterQueue.Send(compensationEvent)
}
```

### 3. **How do you implement Saga with timeouts and retries?**

**Answer:**
Timeouts and retries are implemented at multiple levels:

1. **Step-Level Timeout**:

```go
func (o *SagaOrchestrator) executeStep(ctx context.Context, step SagaStep, sagaCtx *SagaContext) error {
    // Create timeout context for the step
    stepCtx, cancel := context.WithTimeout(ctx, step.GetTimeout())
    defer cancel()

    // Execute with timeout
    return step.Execute(stepCtx, sagaCtx)
}
```

2. **Step-Level Retry**:

```go
func (o *SagaOrchestrator) executeStepWithRetry(ctx context.Context, step SagaStep, sagaCtx *SagaContext) error {
    return o.retryPolicy.Execute(func() error {
        return o.executeStep(ctx, step, sagaCtx)
    })
}
```

3. **Saga-Level Timeout**:

```go
func (o *SagaOrchestrator) ExecuteSaga(ctx context.Context, saga *PaymentSaga) error {
    // Create timeout context for entire saga
    sagaCtx, cancel := context.WithTimeout(ctx, time.Hour) // 1 hour total timeout
    defer cancel()

    return o.executeSagaSteps(sagaCtx, saga)
}
```

4. **Exponential Backoff**:

```go
type RetryPolicy struct {
    MaxAttempts int
    BaseDelay   time.Duration
    MaxDelay    time.Duration
    Multiplier  float64
}

func (r *RetryPolicy) Execute(fn func() error) error {
    delay := r.BaseDelay

    for attempt := 1; attempt <= r.MaxAttempts; attempt++ {
        if err := fn(); err != nil {
            if attempt == r.MaxAttempts {
                return err
            }

            time.Sleep(delay)
            delay = time.Duration(float64(delay) * r.Multiplier)
            if delay > r.MaxDelay {
                delay = r.MaxDelay
            }
        } else {
            return nil
        }
    }

    return fmt.Errorf("max attempts exceeded")
}
```

### 4. **How do you implement Saga orchestration vs choreography?**

**Answer:**

**Orchestration (Centralized Control):**

```go
type SagaOrchestrator struct {
    steps []SagaStep
}

// Central coordinator controls all steps
func (o *SagaOrchestrator) ExecuteSaga(saga *PaymentSaga) error {
    for i, step := range saga.Steps {
        if err := step.Execute(saga.Context); err != nil {
            // Compensate previous steps
            for j := i - 1; j >= 0; j-- {
                if compensatable, ok := saga.Steps[j].(CompensatableStep); ok {
                    compensatable.Compensate(saga.Context)
                }
            }
            return err
        }
    }
    return nil
}
```

**Choreography (Distributed Control):**

```go
// Each service knows what to do based on events
type PaymentService struct {
    eventBus EventBus
}

func (p *PaymentService) HandlePaymentInitiated(event PaymentInitiatedEvent) {
    if err := p.processPayment(event); err != nil {
        // Publish failure event
        p.eventBus.Publish(PaymentFailedEvent{
            SagaID:    event.SagaID,
            PaymentID: event.PaymentID,
            Error:     err.Error(),
        })
    } else {
        // Publish success event
        p.eventBus.Publish(PaymentSuccessEvent{
            SagaID:       event.SagaID,
            PaymentID:    event.PaymentID,
            TransactionID: transactionID,
        })
    }
}

type MerchantService struct {
    eventBus EventBus
}

func (m *MerchantService) HandlePaymentSuccess(event PaymentSuccessEvent) {
    if err := m.creditMerchant(event); err != nil {
        // Publish compensation event
        m.eventBus.Publish(CompensatePaymentEvent{
            SagaID:        event.SagaID,
            TransactionID: event.TransactionID,
        })
    } else {
        // Continue to next step
        m.eventBus.Publish(MerchantCreditedEvent{
            SagaID:     event.SagaID,
            MerchantID: event.MerchantID,
        })
    }
}
```

**Trade-offs:**

- **Orchestration**: Easier to understand and debug, but single point of failure
- **Choreography**: More resilient and scalable, but harder to track and debug

### 5. **How do you ensure Saga isolation and consistency?**

**Answer:**
Saga isolation and consistency are achieved through several techniques:

1. **Semantic Locks**: Use business-level locks instead of database locks

```go
type SemanticsLockService struct {
    locks map[string]time.Time
    mu    sync.RWMutex
}

func (s *SemanticsLockService) AcquireLock(resource string, duration time.Duration) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    if expiry, exists := s.locks[resource]; exists && time.Now().Before(expiry) {
        return fmt.Errorf("resource is locked: %s", resource)
    }

    s.locks[resource] = time.Now().Add(duration)
    return nil
}
```

2. **Versioning**: Use optimistic locking with versions

```go
type VersionedEntity struct {
    ID      string `json:"id"`
    Version int    `json:"version"`
    Data    string `json:"data"`
}

func (s *EntityService) UpdateWithVersion(entity *VersionedEntity) error {
    result, err := s.db.Exec(`
        UPDATE entities
        SET data = ?, version = version + 1
        WHERE id = ? AND version = ?
    `, entity.Data, entity.ID, entity.Version)

    if err != nil {
        return err
    }

    rowsAffected, _ := result.RowsAffected()
    if rowsAffected == 0 {
        return fmt.Errorf("optimistic lock failed - entity was modified")
    }

    return nil
}
```

3. **Commutative Operations**: Design operations to be commutative

```go
// Non-commutative (order matters)
func (a *Account) SetBalance(amount decimal.Decimal) {
    a.Balance = amount // Overwrites - order matters
}

// Commutative (order doesn't matter)
func (a *Account) AdjustBalance(delta decimal.Decimal) {
    a.Balance = a.Balance.Add(delta) // Addition is commutative
}
```

4. **Compensating Transactions**: Make compensation idempotent

```go
func (s *ReservationService) CompensateReservation(reservationID string) error {
    // Idempotent compensation
    reservation, err := s.getReservation(reservationID)
    if err != nil || reservation.Status == "RELEASED" {
        return nil // Already compensated
    }

    return s.releaseReservation(reservationID)
}
```
