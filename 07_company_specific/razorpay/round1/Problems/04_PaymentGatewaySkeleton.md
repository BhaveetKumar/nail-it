---
# Auto-generated front matter
Title: 04 Paymentgatewayskeleton
LastUpdated: 2025-11-06T20:45:58.503299
Tags: []
Status: draft
---

# 04. Payment Gateway Skeleton - Core Payment Processing System

## Title & Summary
Design and implement a payment gateway skeleton that handles payment processing, transaction management, and integration with multiple payment providers with idempotency and fraud detection.

## Problem Statement

Build a payment gateway system that:

1. **Payment Processing**: Handle credit card, debit card, and digital wallet payments
2. **Transaction Management**: Track payment states and handle reversals
3. **Provider Integration**: Connect to multiple payment providers (Stripe, PayPal, etc.)
4. **Idempotency**: Ensure duplicate payment prevention
5. **Fraud Detection**: Basic fraud detection and risk assessment
6. **Webhook Handling**: Process payment status updates from providers

## Requirements & Constraints

### Functional Requirements
- Process payments through multiple providers
- Handle payment reversals and refunds
- Implement idempotent payment requests
- Basic fraud detection and risk scoring
- Webhook processing for payment updates
- Transaction history and reporting

### Non-Functional Requirements
- **Latency**: < 2s for payment processing
- **Consistency**: Strong consistency for transaction state
- **Memory**: Support 100K concurrent transactions
- **Scalability**: Handle 1M transactions per day
- **Reliability**: 99.99% transaction success rate

## API / Interfaces

### REST Endpoints

```go
// Payment Processing
POST   /api/payments/process
GET    /api/payments/{paymentID}
POST   /api/payments/{paymentID}/refund
POST   /api/payments/{paymentID}/reverse

// Transaction Management
GET    /api/transactions
GET    /api/transactions/{transactionID}
POST   /api/transactions/{transactionID}/status

// Provider Management
GET    /api/providers
POST   /api/providers/{providerID}/webhook

// Fraud Detection
POST   /api/fraud/check
GET    /api/fraud/score/{transactionID}

// Webhook
POST   /api/webhooks/{providerID}
```

### Request/Response Examples

```json
// Process Payment
POST /api/payments/process
{
  "amount": 100.00,
  "currency": "USD",
  "paymentMethod": {
    "type": "card",
    "cardNumber": "4111111111111111",
    "expiryMonth": 12,
    "expiryYear": 2025,
    "cvv": "123",
    "holderName": "John Doe"
  },
  "billingAddress": {
    "street": "123 Main St",
    "city": "San Francisco",
    "state": "CA",
    "zipCode": "94105",
    "country": "US"
  },
  "metadata": {
    "orderID": "order123",
    "customerID": "customer456"
  }
}

// Payment Response
{
  "paymentID": "pay_789",
  "status": "processing",
  "amount": 100.00,
  "currency": "USD",
  "provider": "stripe",
  "transactionID": "txn_abc123",
  "createdAt": "2024-01-15T10:30:00Z",
  "fraudScore": 0.2
}
```

## Data Model

### Core Entities

```go
type Payment struct {
    ID              string        `json:"id"`
    Amount          float64       `json:"amount"`
    Currency        string        `json:"currency"`
    Status          PaymentStatus `json:"status"`
    PaymentMethod   PaymentMethod `json:"paymentMethod"`
    BillingAddress  Address       `json:"billingAddress"`
    Provider        string        `json:"provider"`
    ProviderTxnID   string        `json:"providerTxnID"`
    FraudScore      float64       `json:"fraudScore"`
    Metadata        map[string]string `json:"metadata"`
    CreatedAt       time.Time     `json:"createdAt"`
    UpdatedAt       time.Time     `json:"updatedAt"`
    ProcessedAt     *time.Time    `json:"processedAt,omitempty"`
}

type PaymentMethod struct {
    Type        string `json:"type"`
    CardNumber  string `json:"cardNumber,omitempty"`
    ExpiryMonth int    `json:"expiryMonth,omitempty"`
    ExpiryYear  int    `json:"expiryYear,omitempty"`
    CVV         string `json:"cvv,omitempty"`
    HolderName  string `json:"holderName,omitempty"`
    WalletID    string `json:"walletID,omitempty"`
}

type Address struct {
    Street  string `json:"street"`
    City    string `json:"city"`
    State   string `json:"state"`
    ZipCode string `json:"zipCode"`
    Country string `json:"country"`
}

type Transaction struct {
    ID            string            `json:"id"`
    PaymentID     string            `json:"paymentID"`
    Type          TransactionType   `json:"type"`
    Status        TransactionStatus `json:"status"`
    Amount        float64           `json:"amount"`
    Currency      string            `json:"currency"`
    Provider      string            `json:"provider"`
    ProviderTxnID string            `json:"providerTxnID"`
    Reason        string            `json:"reason,omitempty"`
    CreatedAt     time.Time         `json:"createdAt"`
    UpdatedAt     time.Time         `json:"updatedAt"`
}

type Provider struct {
    ID          string            `json:"id"`
    Name        string            `json:"name"`
    Type        string            `json:"type"`
    Config      map[string]string `json:"config"`
    IsActive    bool              `json:"isActive"`
    Priority    int               `json:"priority"`
    CreatedAt   time.Time         `json:"createdAt"`
}

type FraudCheck struct {
    TransactionID string    `json:"transactionID"`
    Score         float64   `json:"score"`
    RiskLevel     string    `json:"riskLevel"`
    Factors       []string  `json:"factors"`
    CreatedAt     time.Time `json:"createdAt"`
}
```

## Approach Overview

### Simple Solution (MVP)
1. In-memory storage with basic transaction tracking
2. Single payment provider integration
3. Simple fraud detection rules
4. Basic webhook handling

### Production-Ready Design
1. **Microservices Architecture**: Separate services for payments, fraud, providers
2. **Event-Driven**: Use message queues for payment processing
3. **Provider Abstraction**: Plugin-based provider integration
4. **Fraud Detection**: ML-based risk scoring
5. **Idempotency**: Redis-based idempotency keys
6. **Audit Trail**: Complete transaction logging

## Detailed Design

### Modular Decomposition

```go
paymentgateway/
├── payments/      # Payment processing
├── providers/     # Payment provider integration
├── fraud/         # Fraud detection
├── webhooks/      # Webhook handling
├── transactions/  # Transaction management
├── idempotency/   # Idempotency handling
└── audit/         # Audit logging
```

### Concurrency Model

```go
type PaymentService struct {
    payments      map[string]*Payment
    transactions  map[string]*Transaction
    providers     map[string]*Provider
    fraudChecks   map[string]*FraudCheck
    idempotency   *IdempotencyService
    mutex         sync.RWMutex
    paymentChan   chan PaymentRequest
    webhookChan   chan WebhookEvent
    auditChan     chan AuditEvent
}

// Goroutines for:
// 1. Payment processing
// 2. Webhook handling
// 3. Fraud detection
// 4. Audit logging
```

### Persistence Strategy

```go
// Database schema
CREATE TABLE payments (
    id VARCHAR(36) PRIMARY KEY,
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    status VARCHAR(20) NOT NULL,
    provider VARCHAR(50),
    provider_txn_id VARCHAR(100),
    fraud_score DECIMAL(3,2),
    metadata JSON,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE transactions (
    id VARCHAR(36) PRIMARY KEY,
    payment_id VARCHAR(36) NOT NULL,
    type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    provider VARCHAR(50),
    provider_txn_id VARCHAR(100),
    created_at TIMESTAMP
);
```

## Optimal Golang Implementation

```go
package main

import (
    "context"
    "crypto/md5"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"

    "github.com/google/uuid"
)

type PaymentStatus string
const (
    PaymentPending   PaymentStatus = "pending"
    PaymentProcessing PaymentStatus = "processing"
    PaymentSuccess   PaymentStatus = "success"
    PaymentFailed    PaymentStatus = "failed"
    PaymentCancelled PaymentStatus = "cancelled"
)

type TransactionType string
const (
    TransactionPayment TransactionType = "payment"
    TransactionRefund  TransactionType = "refund"
    TransactionReverse TransactionType = "reverse"
)

type TransactionStatus string
const (
    TransactionPending   TransactionStatus = "pending"
    TransactionSuccess   TransactionStatus = "success"
    TransactionFailed    TransactionStatus = "failed"
    TransactionCancelled TransactionStatus = "cancelled"
)

type Payment struct {
    ID              string        `json:"id"`
    Amount          float64       `json:"amount"`
    Currency        string        `json:"currency"`
    Status          PaymentStatus `json:"status"`
    PaymentMethod   PaymentMethod `json:"paymentMethod"`
    BillingAddress  Address       `json:"billingAddress"`
    Provider        string        `json:"provider"`
    ProviderTxnID   string        `json:"providerTxnID"`
    FraudScore      float64       `json:"fraudScore"`
    Metadata        map[string]string `json:"metadata"`
    CreatedAt       time.Time     `json:"createdAt"`
    UpdatedAt       time.Time     `json:"updatedAt"`
    ProcessedAt     *time.Time    `json:"processedAt,omitempty"`
}

type PaymentMethod struct {
    Type        string `json:"type"`
    CardNumber  string `json:"cardNumber,omitempty"`
    ExpiryMonth int    `json:"expiryMonth,omitempty"`
    ExpiryYear  int    `json:"expiryYear,omitempty"`
    CVV         string `json:"cvv,omitempty"`
    HolderName  string `json:"holderName,omitempty"`
    WalletID    string `json:"walletID,omitempty"`
}

type Address struct {
    Street  string `json:"street"`
    City    string `json:"city"`
    State   string `json:"state"`
    ZipCode string `json:"zipCode"`
    Country string `json:"country"`
}

type Transaction struct {
    ID            string            `json:"id"`
    PaymentID     string            `json:"paymentID"`
    Type          TransactionType   `json:"type"`
    Status        TransactionStatus `json:"status"`
    Amount        float64           `json:"amount"`
    Currency      string            `json:"currency"`
    Provider      string            `json:"provider"`
    ProviderTxnID string            `json:"providerTxnID"`
    Reason        string            `json:"reason,omitempty"`
    CreatedAt     time.Time         `json:"createdAt"`
    UpdatedAt     time.Time         `json:"updatedAt"`
}

type Provider struct {
    ID          string            `json:"id"`
    Name        string            `json:"name"`
    Type        string            `json:"type"`
    Config      map[string]string `json:"config"`
    IsActive    bool              `json:"isActive"`
    Priority    int               `json:"priority"`
    CreatedAt   time.Time         `json:"createdAt"`
}

type FraudCheck struct {
    TransactionID string    `json:"transactionID"`
    Score         float64   `json:"score"`
    RiskLevel     string    `json:"riskLevel"`
    Factors       []string  `json:"factors"`
    CreatedAt     time.Time `json:"createdAt"`
}

type PaymentRequest struct {
    Amount         float64       `json:"amount"`
    Currency       string        `json:"currency"`
    PaymentMethod  PaymentMethod `json:"paymentMethod"`
    BillingAddress Address       `json:"billingAddress"`
    Metadata       map[string]string `json:"metadata"`
    IdempotencyKey string       `json:"idempotencyKey,omitempty"`
}

type WebhookEvent struct {
    ProviderID    string                 `json:"providerID"`
    EventType     string                 `json:"eventType"`
    TransactionID string                 `json:"transactionID"`
    Data          map[string]interface{} `json:"data"`
    Timestamp     time.Time              `json:"timestamp"`
}

type IdempotencyService struct {
    keys map[string]string
    mutex sync.RWMutex
}

type PaymentService struct {
    payments      map[string]*Payment
    transactions  map[string]*Transaction
    providers     map[string]*Provider
    fraudChecks   map[string]*FraudCheck
    idempotency   *IdempotencyService
    mutex         sync.RWMutex
    paymentChan   chan PaymentRequest
    webhookChan   chan WebhookEvent
    auditChan     chan AuditEvent
}

type AuditEvent struct {
    ID        string    `json:"id"`
    Type      string    `json:"type"`
    EntityID  string    `json:"entityID"`
    Action    string    `json:"action"`
    Data      interface{} `json:"data"`
    Timestamp time.Time `json:"timestamp"`
}

func NewPaymentService() *PaymentService {
    return &PaymentService{
        payments:     make(map[string]*Payment),
        transactions: make(map[string]*Transaction),
        providers:    make(map[string]*Provider),
        fraudChecks:  make(map[string]*FraudCheck),
        idempotency:  &IdempotencyService{keys: make(map[string]string)},
        paymentChan:  make(chan PaymentRequest, 1000),
        webhookChan:  make(chan WebhookEvent, 1000),
        auditChan:    make(chan AuditEvent, 1000),
    }
}

func (ps *PaymentService) ProcessPayment(req PaymentRequest) (*Payment, error) {
    // Check idempotency
    if req.IdempotencyKey != "" {
        if existingPaymentID, exists := ps.idempotency.Get(req.IdempotencyKey); exists {
            return ps.GetPayment(existingPaymentID)
        }
    }

    // Validate payment request
    if err := ps.validatePaymentRequest(req); err != nil {
        return nil, err
    }

    // Create payment
    payment := &Payment{
        ID:             uuid.New().String(),
        Amount:         req.Amount,
        Currency:       req.Currency,
        Status:         PaymentPending,
        PaymentMethod:  req.PaymentMethod,
        BillingAddress: req.BillingAddress,
        Metadata:       req.Metadata,
        CreatedAt:      time.Now(),
        UpdatedAt:      time.Now(),
    }

    // Store payment
    ps.mutex.Lock()
    ps.payments[payment.ID] = payment
    ps.mutex.Unlock()

    // Store idempotency key
    if req.IdempotencyKey != "" {
        ps.idempotency.Set(req.IdempotencyKey, payment.ID)
    }

    // Send to processing channel
    ps.paymentChan <- req

    // Log audit event
    ps.auditChan <- AuditEvent{
        ID:        uuid.New().String(),
        Type:      "payment",
        EntityID:  payment.ID,
        Action:    "created",
        Data:      payment,
        Timestamp: time.Now(),
    }

    return payment, nil
}

func (ps *PaymentService) GetPayment(paymentID string) (*Payment, error) {
    ps.mutex.RLock()
    defer ps.mutex.RUnlock()

    payment, exists := ps.payments[paymentID]
    if !exists {
        return nil, fmt.Errorf("payment not found")
    }

    return payment, nil
}

func (ps *PaymentService) ProcessPaymentAsync(req PaymentRequest) {
    // Fraud detection
    fraudScore := ps.detectFraud(req)
    
    // Select payment provider
    provider := ps.selectProvider(req)
    
    // Process payment with provider
    result := ps.processWithProvider(provider, req)
    
    // Update payment status
    ps.updatePaymentStatus(req, result)
}

func (ps *PaymentService) detectFraud(req PaymentRequest) float64 {
    score := 0.0
    factors := []string{}

    // Check amount
    if req.Amount > 10000 {
        score += 0.3
        factors = append(factors, "high_amount")
    }

    // Check card number (basic validation)
    if len(req.PaymentMethod.CardNumber) < 13 || len(req.PaymentMethod.CardNumber) > 19 {
        score += 0.5
        factors = append(factors, "invalid_card_number")
    }

    // Check expiry date
    if req.PaymentMethod.ExpiryYear < time.Now().Year() ||
       (req.PaymentMethod.ExpiryYear == time.Now().Year() && req.PaymentMethod.ExpiryMonth < int(time.Now().Month())) {
        score += 0.4
        factors = append(factors, "expired_card")
    }

    // Store fraud check
    fraudCheck := &FraudCheck{
        TransactionID: uuid.New().String(),
        Score:         score,
        RiskLevel:     ps.getRiskLevel(score),
        Factors:       factors,
        CreatedAt:     time.Now(),
    }

    ps.mutex.Lock()
    ps.fraudChecks[fraudCheck.TransactionID] = fraudCheck
    ps.mutex.Unlock()

    return score
}

func (ps *PaymentService) getRiskLevel(score float64) string {
    if score < 0.3 {
        return "low"
    } else if score < 0.7 {
        return "medium"
    } else {
        return "high"
    }
}

func (ps *PaymentService) selectProvider(req PaymentRequest) *Provider {
    ps.mutex.RLock()
    defer ps.mutex.RUnlock()

    // Simple provider selection based on priority
    var selectedProvider *Provider
    for _, provider := range ps.providers {
        if provider.IsActive {
            if selectedProvider == nil || provider.Priority < selectedProvider.Priority {
                selectedProvider = provider
            }
        }
    }

    return selectedProvider
}

func (ps *PaymentService) processWithProvider(provider *Provider, req PaymentRequest) map[string]interface{} {
    // Simulate provider API call
    time.Sleep(100 * time.Millisecond)
    
    // Simulate success/failure based on fraud score
    fraudScore := ps.detectFraud(req)
    success := fraudScore < 0.7
    
    result := map[string]interface{}{
        "success": success,
        "transactionID": uuid.New().String(),
        "provider": provider.Name,
        "fraudScore": fraudScore,
    }
    
    if !success {
        result["error"] = "Payment declined due to fraud risk"
    }
    
    return result
}

func (ps *PaymentService) updatePaymentStatus(req PaymentRequest, result map[string]interface{}) {
    // Find payment by amount and metadata
    ps.mutex.Lock()
    defer ps.mutex.Unlock()
    
    for _, payment := range ps.payments {
        if payment.Amount == req.Amount && payment.Status == PaymentPending {
            if result["success"].(bool) {
                payment.Status = PaymentSuccess
                payment.Provider = result["provider"].(string)
                payment.ProviderTxnID = result["transactionID"].(string)
                payment.FraudScore = result["fraudScore"].(float64)
                now := time.Now()
                payment.ProcessedAt = &now
            } else {
                payment.Status = PaymentFailed
                payment.FraudScore = result["fraudScore"].(float64)
            }
            payment.UpdatedAt = time.Now()
            break
        }
    }
}

func (ps *PaymentService) RefundPayment(paymentID string, amount float64, reason string) (*Transaction, error) {
    ps.mutex.Lock()
    defer ps.mutex.Unlock()

    payment, exists := ps.payments[paymentID]
    if !exists {
        return nil, fmt.Errorf("payment not found")
    }

    if payment.Status != PaymentSuccess {
        return nil, fmt.Errorf("payment not successful")
    }

    if amount > payment.Amount {
        return nil, fmt.Errorf("refund amount exceeds payment amount")
    }

    transaction := &Transaction{
        ID:            uuid.New().String(),
        PaymentID:     paymentID,
        Type:          TransactionRefund,
        Status:        TransactionPending,
        Amount:        amount,
        Currency:      payment.Currency,
        Provider:      payment.Provider,
        Reason:        reason,
        CreatedAt:     time.Now(),
        UpdatedAt:     time.Now(),
    }

    ps.transactions[transaction.ID] = transaction

    // Simulate refund processing
    go func() {
        time.Sleep(200 * time.Millisecond)
        ps.mutex.Lock()
        transaction.Status = TransactionSuccess
        transaction.UpdatedAt = time.Now()
        ps.mutex.Unlock()
    }()

    return transaction, nil
}

func (ps *PaymentService) HandleWebhook(providerID string, event WebhookEvent) error {
    ps.webhookChan <- event
    
    // Log audit event
    ps.auditChan <- AuditEvent{
        ID:        uuid.New().String(),
        Type:      "webhook",
        EntityID:  event.TransactionID,
        Action:    "received",
        Data:      event,
        Timestamp: time.Now(),
    }
    
    return nil
}

func (ps *PaymentService) ProcessWebhooks() {
    for event := range ps.webhookChan {
        // Process webhook event
        log.Printf("Processing webhook: %+v", event)
        
        // Update payment status based on webhook
        ps.mutex.Lock()
        for _, payment := range ps.payments {
            if payment.ProviderTxnID == event.TransactionID {
                // Update payment status based on event type
                switch event.EventType {
                case "payment.succeeded":
                    payment.Status = PaymentSuccess
                case "payment.failed":
                    payment.Status = PaymentFailed
                case "payment.cancelled":
                    payment.Status = PaymentCancelled
                }
                payment.UpdatedAt = time.Now()
                break
            }
        }
        ps.mutex.Unlock()
    }
}

func (ps *PaymentService) ProcessAuditLogs() {
    for event := range ps.auditChan {
        // Log audit event (in production, send to audit service)
        log.Printf("Audit: %+v", event)
    }
}

func (ps *PaymentService) validatePaymentRequest(req PaymentRequest) error {
    if req.Amount <= 0 {
        return fmt.Errorf("invalid amount")
    }
    
    if req.Currency == "" {
        return fmt.Errorf("currency required")
    }
    
    if req.PaymentMethod.Type == "" {
        return fmt.Errorf("payment method type required")
    }
    
    return nil
}

// IdempotencyService methods
func (is *IdempotencyService) Set(key, value string) {
    is.mutex.Lock()
    defer is.mutex.Unlock()
    is.keys[key] = value
}

func (is *IdempotencyService) Get(key string) (string, bool) {
    is.mutex.RLock()
    defer is.mutex.RUnlock()
    value, exists := is.keys[key]
    return value, exists
}

func (is *IdempotencyService) GenerateKey(req PaymentRequest) string {
    data := fmt.Sprintf("%.2f%s%s%s", req.Amount, req.Currency, req.PaymentMethod.Type, req.Metadata["orderID"])
    hash := md5.Sum([]byte(data))
    return fmt.Sprintf("%x", hash)
}

// HTTP Handlers
func (ps *PaymentService) ProcessPaymentHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var req PaymentRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    // Generate idempotency key if not provided
    if req.IdempotencyKey == "" {
        req.IdempotencyKey = ps.idempotency.GenerateKey(req)
    }

    payment, err := ps.ProcessPayment(req)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(payment)
}

func (ps *PaymentService) GetPaymentHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodGet {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    paymentID := r.URL.Path[len("/api/payments/"):]
    payment, err := ps.GetPayment(paymentID)
    if err != nil {
        http.Error(w, err.Error(), http.StatusNotFound)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(payment)
}

func (ps *PaymentService) RefundPaymentHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    paymentID := r.URL.Path[len("/api/payments/") : len(r.URL.Path)-len("/refund")]
    
    var req struct {
        Amount float64 `json:"amount"`
        Reason string  `json:"reason"`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    transaction, err := ps.RefundPayment(paymentID, req.Amount, req.Reason)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(transaction)
}

func (ps *PaymentService) WebhookHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    providerID := r.URL.Path[len("/api/webhooks/"):]
    
    var event WebhookEvent
    if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
        http.Error(w, "Invalid webhook", http.StatusBadRequest)
        return
    }

    event.ProviderID = providerID
    event.Timestamp = time.Now()

    if err := ps.HandleWebhook(providerID, event); err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    w.WriteHeader(http.StatusOK)
}

func main() {
    service := NewPaymentService()

    // Initialize providers
    service.providers["stripe"] = &Provider{
        ID:       "stripe",
        Name:     "Stripe",
        Type:     "card",
        IsActive: true,
        Priority: 1,
        CreatedAt: time.Now(),
    }

    // Start background workers
    go service.ProcessPaymentAsync(PaymentRequest{})
    go service.ProcessWebhooks()
    go service.ProcessAuditLogs()

    // HTTP routes
    http.HandleFunc("/api/payments/process", service.ProcessPaymentHandler)
    http.HandleFunc("/api/payments/", service.GetPaymentHandler)
    http.HandleFunc("/api/payments/", service.RefundPaymentHandler)
    http.HandleFunc("/api/webhooks/", service.WebhookHandler)

    log.Println("Payment gateway service starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

## Unit Tests

```go
func TestPaymentService_ProcessPayment(t *testing.T) {
    service := NewPaymentService()

    req := PaymentRequest{
        Amount:   100.00,
        Currency: "USD",
        PaymentMethod: PaymentMethod{
            Type:        "card",
            CardNumber:  "4111111111111111",
            ExpiryMonth: 12,
            ExpiryYear:  2025,
            CVV:         "123",
            HolderName:  "John Doe",
        },
        BillingAddress: Address{
            Street:  "123 Main St",
            City:    "San Francisco",
            State:   "CA",
            ZipCode: "94105",
            Country: "US",
        },
        Metadata: map[string]string{
            "orderID": "order123",
        },
    }

    payment, err := service.ProcessPayment(req)
    if err != nil {
        t.Fatalf("ProcessPayment() error = %v", err)
    }

    if payment.Amount != req.Amount {
        t.Errorf("ProcessPayment() amount = %v, want %v", payment.Amount, req.Amount)
    }

    if payment.Currency != req.Currency {
        t.Errorf("ProcessPayment() currency = %v, want %v", payment.Currency, req.Currency)
    }

    if payment.Status != PaymentPending {
        t.Errorf("ProcessPayment() status = %v, want %v", payment.Status, PaymentPending)
    }
}

func TestPaymentService_Idempotency(t *testing.T) {
    service := NewPaymentService()

    req := PaymentRequest{
        Amount:   100.00,
        Currency: "USD",
        PaymentMethod: PaymentMethod{
            Type: "card",
        },
        IdempotencyKey: "test-key-123",
    }

    // First payment
    payment1, err := service.ProcessPayment(req)
    if err != nil {
        t.Fatalf("ProcessPayment() error = %v", err)
    }

    // Second payment with same idempotency key
    payment2, err := service.ProcessPayment(req)
    if err != nil {
        t.Fatalf("ProcessPayment() error = %v", err)
    }

    if payment1.ID != payment2.ID {
        t.Errorf("Idempotency failed: payment1.ID = %v, payment2.ID = %v", payment1.ID, payment2.ID)
    }
}

func TestPaymentService_RefundPayment(t *testing.T) {
    service := NewPaymentService()

    // Create a successful payment
    payment := &Payment{
        ID:       "pay123",
        Amount:   100.00,
        Currency: "USD",
        Status:   PaymentSuccess,
    }
    service.payments[payment.ID] = payment

    // Refund payment
    transaction, err := service.RefundPayment(payment.ID, 50.00, "Customer request")
    if err != nil {
        t.Fatalf("RefundPayment() error = %v", err)
    }

    if transaction.Type != TransactionRefund {
        t.Errorf("RefundPayment() type = %v, want %v", transaction.Type, TransactionRefund)
    }

    if transaction.Amount != 50.00 {
        t.Errorf("RefundPayment() amount = %v, want 50.00", transaction.Amount)
    }
}
```

## Complexity Analysis

### Time Complexity
- **Process Payment**: O(1) - Hash map operations
- **Fraud Detection**: O(1) - Simple rule-based checks
- **Provider Selection**: O(P) - Linear scan through providers
- **Webhook Processing**: O(1) - Hash map lookup

### Space Complexity
- **Payment Storage**: O(P) where P is number of payments
- **Transaction Storage**: O(T) where T is number of transactions
- **Idempotency Storage**: O(I) where I is number of idempotency keys
- **Total**: O(P + T + I)

## Edge Cases & Validation

### Input Validation
- Invalid payment amounts (negative, zero)
- Invalid currency codes
- Invalid card numbers and expiry dates
- Missing required fields
- Invalid billing addresses

### Error Scenarios
- Payment provider failures
- Network timeouts
- Fraud detection failures
- Webhook processing errors
- Idempotency key conflicts

### Boundary Conditions
- Maximum payment amount limits
- Minimum payment amount limits
- Card number length validation
- Expiry date validation
- CVV format validation

## Extension Ideas (Scaling)

### Horizontal Scaling
1. **Load Balancing**: Multiple service instances
2. **Database Sharding**: Partition by payment ID
3. **Message Queue**: Kafka for payment processing
4. **Cache Clustering**: Redis cluster for idempotency

### Performance Optimization
1. **Payment Caching**: Redis for payment status
2. **Provider Pooling**: Connection pooling for providers
3. **Async Processing**: Background payment processing
4. **Batch Processing**: Batch webhook processing

### Advanced Features
1. **Machine Learning**: Advanced fraud detection
2. **Multi-currency**: Currency conversion and support
3. **Payment Plans**: Installment and subscription payments
4. **Analytics**: Payment analytics and reporting

## 20 Follow-up Questions

### 1. How would you handle payment provider failures?
**Answer**: Implement circuit breaker pattern with fallback providers. Use retry logic with exponential backoff. Implement provider health checks and automatic failover. Consider using multiple providers for redundancy.

### 2. What's your strategy for handling high-value transactions?
**Answer**: Implement additional fraud checks for high-value transactions. Use manual review for transactions above certain thresholds. Implement additional verification steps. Consider using specialized high-risk providers.

### 3. How do you ensure PCI compliance?
**Answer**: Use tokenization for card data storage. Implement encryption for sensitive data. Use certified payment processors. Implement access controls and audit logging. Consider using PCI-compliant infrastructure.

### 4. What happens if a webhook is missed?
**Answer**: Implement webhook retry mechanisms with exponential backoff. Use polling as fallback for critical status updates. Implement webhook signature verification. Consider using message queues for reliable delivery.

### 5. How would you implement payment reconciliation?
**Answer**: Implement daily reconciliation with payment providers. Use transaction matching algorithms. Implement discrepancy detection and reporting. Consider using automated reconciliation tools.

### 6. What's your approach to handling chargebacks?
**Answer**: Implement chargeback notification system. Use dispute management workflows. Implement evidence collection and submission. Consider using chargeback prevention tools.

### 7. How do you handle payment reversals?
**Answer**: Implement immediate reversal for failed payments. Use provider-specific reversal APIs. Implement reversal tracking and reporting. Consider using automated reversal processing.

### 8. What's your strategy for handling payment disputes?
**Answer**: Implement dispute management system. Use evidence collection and submission. Implement dispute tracking and reporting. Consider using automated dispute resolution.

### 9. How would you implement payment analytics?
**Answer**: Use data warehouse for payment analytics. Implement real-time dashboards. Use machine learning for payment insights. Consider using business intelligence tools.

### 10. What's your approach to handling payment fraud?
**Answer**: Implement multi-layer fraud detection. Use machine learning for fraud patterns. Implement real-time fraud scoring. Consider using third-party fraud services.

### 11. How do you handle payment retries?
**Answer**: Implement retry logic with exponential backoff. Use different providers for retries. Implement retry limits and timeouts. Consider using intelligent retry strategies.

### 12. What's your strategy for handling payment timeouts?
**Answer**: Implement timeout handling with fallback providers. Use async processing for long-running payments. Implement timeout monitoring and alerting. Consider using circuit breakers.

### 13. How would you implement payment batching?
**Answer**: Implement batch payment processing. Use message queues for batch operations. Implement batch reconciliation. Consider using batch optimization algorithms.

### 14. What's your approach to handling payment refunds?
**Answer**: Implement automated refund processing. Use provider-specific refund APIs. Implement refund tracking and reporting. Consider using refund optimization strategies.

### 15. How do you handle payment notifications?
**Answer**: Implement multi-channel notifications. Use event-driven architecture for notifications. Implement notification templates and personalization. Consider using notification optimization.

### 16. What's your strategy for handling payment data retention?
**Answer**: Implement data retention policies. Use data archiving for old payments. Implement data anonymization for privacy. Consider using data lifecycle management.

### 17. How would you implement payment testing?
**Answer**: Use sandbox environments for testing. Implement test data management. Use automated testing for payment flows. Consider using payment simulation tools.

### 18. What's your approach to handling payment security?
**Answer**: Implement encryption for sensitive data. Use secure communication protocols. Implement access controls and authentication. Consider using security monitoring tools.

### 19. How do you handle payment performance monitoring?
**Answer**: Implement performance monitoring and alerting. Use APM tools for payment tracking. Implement SLA monitoring. Consider using performance optimization tools.

### 20. What's your strategy for handling payment compliance?
**Answer**: Implement compliance monitoring and reporting. Use automated compliance checks. Implement audit trails and logging. Consider using compliance management tools.

## Evaluation Checklist

### Code Quality (25%)
- [ ] Clean, readable Go code with proper error handling
- [ ] Appropriate use of interfaces and structs
- [ ] Proper concurrency patterns (goroutines, channels)
- [ ] Good separation of concerns

### Architecture (25%)
- [ ] Scalable design with provider abstraction
- [ ] Proper payment state management
- [ ] Efficient fraud detection
- [ ] Idempotency handling

### Functionality (25%)
- [ ] Payment processing working
- [ ] Provider integration functional
- [ ] Fraud detection implemented
- [ ] Webhook handling working

### Testing (15%)
- [ ] Unit tests for core functionality
- [ ] Integration tests for API endpoints
- [ ] Edge case testing
- [ ] Performance testing

### Discussion (10%)
- [ ] Clear explanation of design decisions
- [ ] Understanding of payment processing
- [ ] Knowledge of fraud detection
- [ ] Ability to discuss trade-offs

## Discussion Pointers

### Key Points to Highlight
1. **Payment State Management**: Explain the payment lifecycle and state transitions
2. **Provider Abstraction**: Discuss the plugin-based provider integration
3. **Fraud Detection**: Explain the rule-based fraud detection system
4. **Idempotency**: Discuss the importance of idempotent payment requests
5. **Webhook Processing**: Explain the event-driven webhook handling

### Trade-offs to Discuss
1. **Security vs Performance**: Encryption vs processing speed trade-offs
2. **Consistency vs Availability**: Strong consistency vs high availability
3. **Cost vs Reliability**: Provider cost vs reliability trade-offs
4. **Speed vs Accuracy**: Fast processing vs fraud detection accuracy
5. **Simplicity vs Features**: Simple design vs advanced features

### Extension Scenarios
1. **Multi-region Deployment**: How to handle geographic distribution
2. **Advanced Fraud Detection**: Machine learning integration
3. **Payment Analytics**: Real-time dashboards and reporting
4. **Compliance Integration**: PCI DSS and regulatory compliance
5. **Enterprise Features**: Multi-tenant and white-label solutions
