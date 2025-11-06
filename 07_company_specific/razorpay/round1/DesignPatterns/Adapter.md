---
# Auto-generated front matter
Title: Adapter
LastUpdated: 2025-11-06T20:45:58.525011
Tags: []
Status: draft
---

# Adapter Pattern

## Pattern Name & Intent

**Adapter** is a structural design pattern that allows objects with incompatible interfaces to collaborate. It acts as a wrapper between two objects, catching calls for one object and transforming them to format and interface recognizable by the second object.

**Key Intent:**

- Make incompatible interfaces work together
- Allow existing code to work with new libraries or systems
- Convert one interface to another that clients expect
- Enable integration without modifying existing code
- Provide a bridge between legacy and modern systems

## When to Use

**Use Adapter when:**

1. **Legacy Integration**: Need to integrate legacy systems with modern applications
2. **Third-Party Libraries**: Want to use third-party libraries with incompatible interfaces
3. **Interface Mismatch**: Existing classes have useful functionality but wrong interface
4. **Multiple Data Sources**: Need to access different data sources through a common interface
5. **Gradual Migration**: Migrating from old to new systems gradually
6. **API Standardization**: Want to standardize access to multiple external APIs

**Don't use when:**

- Interfaces are already compatible
- You can modify the target interface directly
- The adaptation requires significant data transformation
- Performance overhead is critical

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Gateway Adapter

```go
// Different payment gateways with different interfaces
type RazorpayGateway struct{}
func (r *RazorpayGateway) CreateOrder(amount int, currency string, notes map[string]string) (*RazorpayOrder, error) { /* ... */ }
func (r *RazorpayGateway) CapturePayment(paymentID string, amount int) (*RazorpayPayment, error) { /* ... */ }

type StripeGateway struct{}
func (s *StripeGateway) CreatePaymentIntent(amountInCents int64, curr string, metadata map[string]string) (*StripePaymentIntent, error) { /* ... */ }
func (s *StripeGateway) ConfirmPayment(intentID string) (*StripePayment, error) { /* ... */ }

// Common interface for payment processing
type PaymentProcessor interface {
    ProcessPayment(req PaymentRequest) (*PaymentResponse, error)
    RefundPayment(transactionID string, amount decimal.Decimal) (*RefundResponse, error)
}

// Adapters to make incompatible gateways compatible
type RazorpayAdapter struct {
    gateway *RazorpayGateway
}

func (r *RazorpayAdapter) ProcessPayment(req PaymentRequest) (*PaymentResponse, error) {
    // Convert standard request to Razorpay format
    amount := int(req.Amount.Mul(decimal.NewFromInt(100)).IntPart()) // Convert to paise
    notes := map[string]string{
        "customer_id": req.CustomerID,
        "order_id": req.OrderID,
    }

    order, err := r.gateway.CreateOrder(amount, req.Currency, notes)
    if err != nil {
        return nil, err
    }

    // Convert Razorpay response to standard format
    return &PaymentResponse{
        TransactionID: order.ID,
        Status: "pending",
        Amount: req.Amount,
        Currency: req.Currency,
    }, nil
}
```

### 2. Banking System Integration

```go
// Legacy mainframe banking system
type LegacyBankingSystem struct{}
func (l *LegacyBankingSystem) ACCT_BAL_INQ(acctNum string) (int64, error) { /* ... */ }
func (l *LegacyBankingSystem) FUND_XFER(fromAcct, toAcct string, amt int64) (string, error) { /* ... */ }

// Modern banking interface
type BankingService interface {
    GetAccountBalance(accountID string) (decimal.Decimal, error)
    TransferFunds(from, to string, amount decimal.Decimal) (*TransferResult, error)
}

// Adapter for legacy system
type LegacyBankingAdapter struct {
    legacySystem *LegacyBankingSystem
}

func (l *LegacyBankingAdapter) GetAccountBalance(accountID string) (decimal.Decimal, error) {
    balanceInCents, err := l.legacySystem.ACCT_BAL_INQ(accountID)
    if err != nil {
        return decimal.Zero, err
    }

    // Convert cents to decimal
    return decimal.NewFromInt(balanceInCents).Div(decimal.NewFromInt(100)), nil
}

func (l *LegacyBankingAdapter) TransferFunds(from, to string, amount decimal.Decimal) (*TransferResult, error) {
    // Convert decimal to cents
    amountInCents := amount.Mul(decimal.NewFromInt(100)).IntPart()

    transferID, err := l.legacySystem.FUND_XFER(from, to, amountInCents)
    if err != nil {
        return nil, err
    }

    return &TransferResult{
        TransferID: transferID,
        Status: "completed",
        Amount: amount,
    }, nil
}
```

### 3. Market Data Provider Adapter

```go
// Different market data providers
type BloombergAPI struct{}
func (b *BloombergAPI) GetSecurityPrice(ticker string) (*BloombergQuote, error) { /* ... */ }

type ReutersAPI struct{}
func (r *ReutersAPI) RetrieveMarketData(symbol string) (*ReutersData, error) { /* ... */ }

// Standard market data interface
type MarketDataProvider interface {
    GetQuote(symbol string) (*Quote, error)
    GetHistoricalData(symbol string, from, to time.Time) ([]*HistoricalPrice, error)
}

// Adapters for each provider
type BloombergAdapter struct {
    api *BloombergAPI
}

func (b *BloombergAdapter) GetQuote(symbol string) (*Quote, error) {
    quote, err := b.api.GetSecurityPrice(symbol)
    if err != nil {
        return nil, err
    }

    return &Quote{
        Symbol: symbol,
        Price: quote.LastPrice,
        Bid: quote.BidPrice,
        Ask: quote.AskPrice,
        Timestamp: quote.Timestamp,
    }, nil
}
```

### 4. Notification Service Adapter

```go
// Different notification providers
type TwilioSMS struct{}
func (t *TwilioSMS) SendMessage(to, from, body string) (*TwilioResponse, error) { /* ... */ }

type SendGridEmail struct{}
func (s *SendGridEmail) Send(email *SendGridMail) (*SendGridResponse, error) { /* ... */ }

type FCMPush struct{}
func (f *FCMPush) SendToDevice(deviceToken string, payload *FCMPayload) (*FCMResult, error) { /* ... */ }

// Common notification interface
type NotificationService interface {
    SendNotification(req NotificationRequest) (*NotificationResponse, error)
}

// Adapters for each provider
type TwilioAdapter struct {
    client *TwilioSMS
    fromNumber string
}

func (t *TwilioAdapter) SendNotification(req NotificationRequest) (*NotificationResponse, error) {
    if req.Type != "SMS" {
        return nil, fmt.Errorf("unsupported notification type: %s", req.Type)
    }

    response, err := t.client.SendMessage(req.Recipient, t.fromNumber, req.Message)
    if err != nil {
        return nil, err
    }

    return &NotificationResponse{
        ID: response.SID,
        Status: "sent",
        Provider: "twilio",
    }, nil
}
```

## Go Implementation

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "time"
    "github.com/shopspring/decimal"
)

// Target interface that clients expect
type PaymentProcessor interface {
    ProcessPayment(req PaymentRequest) (*PaymentResponse, error)
    RefundPayment(transactionID string, amount decimal.Decimal) (*RefundResponse, error)
    GetPaymentStatus(transactionID string) (*PaymentStatus, error)
}

// Standard request/response structures
type PaymentRequest struct {
    CustomerID    string          `json:"customer_id"`
    OrderID       string          `json:"order_id"`
    Amount        decimal.Decimal `json:"amount"`
    Currency      string          `json:"currency"`
    PaymentMethod string          `json:"payment_method"`
    Description   string          `json:"description"`
    Metadata      map[string]string `json:"metadata"`
}

type PaymentResponse struct {
    TransactionID string          `json:"transaction_id"`
    Status        string          `json:"status"`
    Amount        decimal.Decimal `json:"amount"`
    Currency      string          `json:"currency"`
    CreatedAt     time.Time       `json:"created_at"`
    PaymentURL    string          `json:"payment_url,omitempty"`
}

type RefundResponse struct {
    RefundID      string          `json:"refund_id"`
    TransactionID string          `json:"transaction_id"`
    Amount        decimal.Decimal `json:"amount"`
    Status        string          `json:"status"`
    CreatedAt     time.Time       `json:"created_at"`
}

type PaymentStatus struct {
    TransactionID string          `json:"transaction_id"`
    Status        string          `json:"status"`
    Amount        decimal.Decimal `json:"amount"`
    Currency      string          `json:"currency"`
    UpdatedAt     time.Time       `json:"updated_at"`
}

// Adaptee 1: Razorpay Gateway (existing system with incompatible interface)
type RazorpayGateway struct {
    keyID     string
    keySecret string
    baseURL   string
}

type RazorpayOrder struct {
    ID          string                 `json:"id"`
    Entity      string                 `json:"entity"`
    Amount      int                    `json:"amount"`      // Amount in paise
    Currency    string                 `json:"currency"`
    Receipt     string                 `json:"receipt"`
    Status      string                 `json:"status"`
    CreatedAt   int64                  `json:"created_at"`
    Notes       map[string]interface{} `json:"notes"`
}

type RazorpayPayment struct {
    ID          string                 `json:"id"`
    Entity      string                 `json:"entity"`
    Amount      int                    `json:"amount"`
    Currency    string                 `json:"currency"`
    Status      string                 `json:"status"`
    OrderID     string                 `json:"order_id"`
    Method      string                 `json:"method"`
    Description string                 `json:"description"`
    CreatedAt   int64                  `json:"created_at"`
    Notes       map[string]interface{} `json:"notes"`
}

type RazorpayRefund struct {
    ID          string `json:"id"`
    Entity      string `json:"entity"`
    Amount      int    `json:"amount"`
    Currency    string `json:"currency"`
    PaymentID   string `json:"payment_id"`
    Status      string `json:"status"`
    CreatedAt   int64  `json:"created_at"`
}

func NewRazorpayGateway(keyID, keySecret string) *RazorpayGateway {
    return &RazorpayGateway{
        keyID:     keyID,
        keySecret: keySecret,
        baseURL:   "https://api.razorpay.com/v1",
    }
}

func (r *RazorpayGateway) CreateOrder(amount int, currency, receipt string, notes map[string]interface{}) (*RazorpayOrder, error) {
    // Simulate Razorpay API call
    order := &RazorpayOrder{
        ID:        fmt.Sprintf("order_%d", time.Now().UnixNano()),
        Entity:    "order",
        Amount:    amount,
        Currency:  currency,
        Receipt:   receipt,
        Status:    "created",
        CreatedAt: time.Now().Unix(),
        Notes:     notes,
    }

    log.Printf("Razorpay: Created order %s for amount %d %s", order.ID, amount, currency)
    return order, nil
}

func (r *RazorpayGateway) CapturePayment(paymentID string, amount int) (*RazorpayPayment, error) {
    // Simulate payment capture
    payment := &RazorpayPayment{
        ID:        paymentID,
        Entity:    "payment",
        Amount:    amount,
        Status:    "captured",
        CreatedAt: time.Now().Unix(),
    }

    log.Printf("Razorpay: Captured payment %s for amount %d", paymentID, amount)
    return payment, nil
}

func (r *RazorpayGateway) RefundPayment(paymentID string, amount int) (*RazorpayRefund, error) {
    // Simulate refund
    refund := &RazorpayRefund{
        ID:        fmt.Sprintf("rfnd_%d", time.Now().UnixNano()),
        Entity:    "refund",
        Amount:    amount,
        PaymentID: paymentID,
        Status:    "processed",
        CreatedAt: time.Now().Unix(),
    }

    log.Printf("Razorpay: Processed refund %s for payment %s", refund.ID, paymentID)
    return refund, nil
}

func (r *RazorpayGateway) FetchPayment(paymentID string) (*RazorpayPayment, error) {
    // Simulate fetching payment
    payment := &RazorpayPayment{
        ID:        paymentID,
        Entity:    "payment",
        Amount:    10000, // 100.00 in paise
        Currency:  "INR",
        Status:    "captured",
        CreatedAt: time.Now().Unix(),
    }

    return payment, nil
}

// Adaptee 2: Stripe Gateway (different incompatible interface)
type StripeGateway struct {
    secretKey string
    baseURL   string
}

type StripePaymentIntent struct {
    ID                string            `json:"id"`
    Object            string            `json:"object"`
    Amount            int64             `json:"amount"`      // Amount in cents
    Currency          string            `json:"currency"`
    Status            string            `json:"status"`
    ClientSecret      string            `json:"client_secret"`
    Description       string            `json:"description"`
    Metadata          map[string]string `json:"metadata"`
    Created           int64             `json:"created"`
}

type StripeRefund struct {
    ID               string `json:"id"`
    Object           string `json:"object"`
    Amount           int64  `json:"amount"`
    Currency         string `json:"currency"`
    PaymentIntentID  string `json:"payment_intent"`
    Status           string `json:"status"`
    Created          int64  `json:"created"`
}

func NewStripeGateway(secretKey string) *StripeGateway {
    return &StripeGateway{
        secretKey: secretKey,
        baseURL:   "https://api.stripe.com/v1",
    }
}

func (s *StripeGateway) CreatePaymentIntent(amount int64, currency string, description string, metadata map[string]string) (*StripePaymentIntent, error) {
    // Simulate Stripe API call
    intent := &StripePaymentIntent{
        ID:           fmt.Sprintf("pi_%d", time.Now().UnixNano()),
        Object:       "payment_intent",
        Amount:       amount,
        Currency:     currency,
        Status:       "requires_payment_method",
        ClientSecret: fmt.Sprintf("pi_%d_secret", time.Now().UnixNano()),
        Description:  description,
        Metadata:     metadata,
        Created:      time.Now().Unix(),
    }

    log.Printf("Stripe: Created payment intent %s for amount %d %s", intent.ID, amount, currency)
    return intent, nil
}

func (s *StripeGateway) ConfirmPaymentIntent(paymentIntentID string) (*StripePaymentIntent, error) {
    // Simulate payment confirmation
    intent := &StripePaymentIntent{
        ID:      paymentIntentID,
        Object:  "payment_intent",
        Status:  "succeeded",
        Created: time.Now().Unix(),
    }

    log.Printf("Stripe: Confirmed payment intent %s", paymentIntentID)
    return intent, nil
}

func (s *StripeGateway) CreateRefund(paymentIntentID string, amount int64) (*StripeRefund, error) {
    // Simulate refund creation
    refund := &StripeRefund{
        ID:              fmt.Sprintf("re_%d", time.Now().UnixNano()),
        Object:          "refund",
        Amount:          amount,
        PaymentIntentID: paymentIntentID,
        Status:          "succeeded",
        Created:         time.Now().Unix(),
    }

    log.Printf("Stripe: Created refund %s for payment intent %s", refund.ID, paymentIntentID)
    return refund, nil
}

func (s *StripeGateway) RetrievePaymentIntent(paymentIntentID string) (*StripePaymentIntent, error) {
    // Simulate retrieving payment intent
    intent := &StripePaymentIntent{
        ID:       paymentIntentID,
        Object:   "payment_intent",
        Amount:   10000, // $100.00 in cents
        Currency: "usd",
        Status:   "succeeded",
        Created:  time.Now().Unix(),
    }

    return intent, nil
}

// Adapter 1: Razorpay Adapter
type RazorpayAdapter struct {
    gateway *RazorpayGateway
}

func NewRazorpayAdapter(gateway *RazorpayGateway) *RazorpayAdapter {
    return &RazorpayAdapter{gateway: gateway}
}

func (r *RazorpayAdapter) ProcessPayment(req PaymentRequest) (*PaymentResponse, error) {
    // Convert decimal amount to paise (Razorpay uses paise)
    amountInPaise := int(req.Amount.Mul(decimal.NewFromInt(100)).IntPart())

    // Convert metadata to interface{} map
    notes := make(map[string]interface{})
    for k, v := range req.Metadata {
        notes[k] = v
    }
    notes["customer_id"] = req.CustomerID
    notes["payment_method"] = req.PaymentMethod

    // Create order using Razorpay API
    order, err := r.gateway.CreateOrder(
        amountInPaise,
        req.Currency,
        req.OrderID,
        notes,
    )
    if err != nil {
        return nil, fmt.Errorf("razorpay order creation failed: %w", err)
    }

    // Convert Razorpay response to standard format
    return &PaymentResponse{
        TransactionID: order.ID,
        Status:        r.convertRazorpayStatus(order.Status),
        Amount:        req.Amount,
        Currency:      req.Currency,
        CreatedAt:     time.Unix(order.CreatedAt, 0),
        PaymentURL:    fmt.Sprintf("https://checkout.razorpay.com/v1/checkout.js?order_id=%s", order.ID),
    }, nil
}

func (r *RazorpayAdapter) RefundPayment(transactionID string, amount decimal.Decimal) (*RefundResponse, error) {
    // Convert amount to paise
    amountInPaise := int(amount.Mul(decimal.NewFromInt(100)).IntPart())

    // Create refund using Razorpay API
    refund, err := r.gateway.RefundPayment(transactionID, amountInPaise)
    if err != nil {
        return nil, fmt.Errorf("razorpay refund failed: %w", err)
    }

    // Convert to standard format
    return &RefundResponse{
        RefundID:      refund.ID,
        TransactionID: refund.PaymentID,
        Amount:        decimal.NewFromInt(int64(refund.Amount)).Div(decimal.NewFromInt(100)),
        Status:        r.convertRazorpayStatus(refund.Status),
        CreatedAt:     time.Unix(refund.CreatedAt, 0),
    }, nil
}

func (r *RazorpayAdapter) GetPaymentStatus(transactionID string) (*PaymentStatus, error) {
    // Fetch payment from Razorpay
    payment, err := r.gateway.FetchPayment(transactionID)
    if err != nil {
        return nil, fmt.Errorf("razorpay payment fetch failed: %w", err)
    }

    // Convert to standard format
    return &PaymentStatus{
        TransactionID: payment.ID,
        Status:        r.convertRazorpayStatus(payment.Status),
        Amount:        decimal.NewFromInt(int64(payment.Amount)).Div(decimal.NewFromInt(100)),
        Currency:      payment.Currency,
        UpdatedAt:     time.Unix(payment.CreatedAt, 0),
    }, nil
}

func (r *RazorpayAdapter) convertRazorpayStatus(status string) string {
    switch status {
    case "created":
        return "pending"
    case "attempted":
        return "processing"
    case "paid", "captured":
        return "completed"
    case "failed":
        return "failed"
    default:
        return status
    }
}

// Adapter 2: Stripe Adapter
type StripeAdapter struct {
    gateway *StripeGateway
}

func NewStripeAdapter(gateway *StripeGateway) *StripeAdapter {
    return &StripeAdapter{gateway: gateway}
}

func (s *StripeAdapter) ProcessPayment(req PaymentRequest) (*PaymentResponse, error) {
    // Convert decimal amount to cents (Stripe uses cents)
    amountInCents := req.Amount.Mul(decimal.NewFromInt(100)).IntPart()

    // Create payment intent using Stripe API
    intent, err := s.gateway.CreatePaymentIntent(
        amountInCents,
        req.Currency,
        req.Description,
        req.Metadata,
    )
    if err != nil {
        return nil, fmt.Errorf("stripe payment intent creation failed: %w", err)
    }

    // Convert Stripe response to standard format
    return &PaymentResponse{
        TransactionID: intent.ID,
        Status:        s.convertStripeStatus(intent.Status),
        Amount:        req.Amount,
        Currency:      req.Currency,
        CreatedAt:     time.Unix(intent.Created, 0),
        PaymentURL:    fmt.Sprintf("https://checkout.stripe.com/pay/%s", intent.ClientSecret),
    }, nil
}

func (s *StripeAdapter) RefundPayment(transactionID string, amount decimal.Decimal) (*RefundResponse, error) {
    // Convert amount to cents
    amountInCents := amount.Mul(decimal.NewFromInt(100)).IntPart()

    // Create refund using Stripe API
    refund, err := s.gateway.CreateRefund(transactionID, amountInCents)
    if err != nil {
        return nil, fmt.Errorf("stripe refund failed: %w", err)
    }

    // Convert to standard format
    return &RefundResponse{
        RefundID:      refund.ID,
        TransactionID: refund.PaymentIntentID,
        Amount:        decimal.NewFromInt(refund.Amount).Div(decimal.NewFromInt(100)),
        Status:        s.convertStripeStatus(refund.Status),
        CreatedAt:     time.Unix(refund.Created, 0),
    }, nil
}

func (s *StripeAdapter) GetPaymentStatus(transactionID string) (*PaymentStatus, error) {
    // Retrieve payment intent from Stripe
    intent, err := s.gateway.RetrievePaymentIntent(transactionID)
    if err != nil {
        return nil, fmt.Errorf("stripe payment intent retrieval failed: %w", err)
    }

    // Convert to standard format
    return &PaymentStatus{
        TransactionID: intent.ID,
        Status:        s.convertStripeStatus(intent.Status),
        Amount:        decimal.NewFromInt(intent.Amount).Div(decimal.NewFromInt(100)),
        Currency:      intent.Currency,
        UpdatedAt:     time.Unix(intent.Created, 0),
    }, nil
}

func (s *StripeAdapter) convertStripeStatus(status string) string {
    switch status {
    case "requires_payment_method", "requires_confirmation":
        return "pending"
    case "processing":
        return "processing"
    case "succeeded":
        return "completed"
    case "requires_action":
        return "action_required"
    case "canceled":
        return "cancelled"
    default:
        return status
    }
}

// Payment Service that uses adapters
type PaymentService struct {
    processors map[string]PaymentProcessor
}

func NewPaymentService() *PaymentService {
    return &PaymentService{
        processors: make(map[string]PaymentProcessor),
    }
}

func (p *PaymentService) RegisterProcessor(name string, processor PaymentProcessor) {
    p.processors[name] = processor
}

func (p *PaymentService) ProcessPayment(provider string, req PaymentRequest) (*PaymentResponse, error) {
    processor, exists := p.processors[provider]
    if !exists {
        return nil, fmt.Errorf("unsupported payment provider: %s", provider)
    }

    return processor.ProcessPayment(req)
}

func (p *PaymentService) RefundPayment(provider string, transactionID string, amount decimal.Decimal) (*RefundResponse, error) {
    processor, exists := p.processors[provider]
    if !exists {
        return nil, fmt.Errorf("unsupported payment provider: %s", provider)
    }

    return processor.RefundPayment(transactionID, amount)
}

func (p *PaymentService) GetPaymentStatus(provider string, transactionID string) (*PaymentStatus, error) {
    processor, exists := p.processors[provider]
    if !exists {
        return nil, fmt.Errorf("unsupported payment provider: %s", provider)
    }

    return processor.GetPaymentStatus(transactionID)
}

func (p *PaymentService) GetSupportedProviders() []string {
    var providers []string
    for name := range p.processors {
        providers = append(providers, name)
    }
    return providers
}

// Example usage
func main() {
    fmt.Println("=== Adapter Pattern Demo ===\n")

    // Create payment gateways
    razorpayGateway := NewRazorpayGateway("rzp_test_key", "rzp_test_secret")
    stripeGateway := NewStripeGateway("sk_test_stripe_key")

    // Create adapters
    razorpayAdapter := NewRazorpayAdapter(razorpayGateway)
    stripeAdapter := NewStripeAdapter(stripeGateway)

    // Create payment service and register adapters
    paymentService := NewPaymentService()
    paymentService.RegisterProcessor("razorpay", razorpayAdapter)
    paymentService.RegisterProcessor("stripe", stripeAdapter)

    // Common payment request
    paymentRequest := PaymentRequest{
        CustomerID:    "CUST_123",
        OrderID:       "ORDER_456",
        Amount:        decimal.NewFromFloat(99.99),
        Currency:      "USD",
        PaymentMethod: "card",
        Description:   "Product purchase",
        Metadata: map[string]string{
            "product_id": "PROD_789",
            "campaign":   "summer_sale",
        },
    }

    // Process payment with different providers
    providers := []string{"razorpay", "stripe"}

    for _, provider := range providers {
        fmt.Printf("=== Processing with %s ===\n", provider)

        // Process payment
        response, err := paymentService.ProcessPayment(provider, paymentRequest)
        if err != nil {
            fmt.Printf("Payment failed: %v\n", err)
            continue
        }

        fmt.Printf("Payment Response:\n")
        responseJSON, _ := json.MarshalIndent(response, "", "  ")
        fmt.Printf("%s\n", responseJSON)

        // Get payment status
        status, err := paymentService.GetPaymentStatus(provider, response.TransactionID)
        if err != nil {
            fmt.Printf("Status check failed: %v\n", err)
        } else {
            fmt.Printf("\nPayment Status:\n")
            statusJSON, _ := json.MarshalIndent(status, "", "  ")
            fmt.Printf("%s\n", statusJSON)
        }

        // Process refund
        refundAmount := decimal.NewFromFloat(29.99)
        refund, err := paymentService.RefundPayment(provider, response.TransactionID, refundAmount)
        if err != nil {
            fmt.Printf("Refund failed: %v\n", err)
        } else {
            fmt.Printf("\nRefund Response:\n")
            refundJSON, _ := json.MarshalIndent(refund, "", "  ")
            fmt.Printf("%s\n", refundJSON)
        }

        fmt.Println()
    }

    // Show supported providers
    fmt.Printf("Supported providers: %v\n", paymentService.GetSupportedProviders())

    // Demonstrate error handling with unsupported provider
    fmt.Println("\n=== Error Handling Demo ===")
    _, err := paymentService.ProcessPayment("unsupported_provider", paymentRequest)
    if err != nil {
        fmt.Printf("Expected error: %v\n", err)
    }

    fmt.Println("\n=== Adapter Pattern Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Object Adapter (Composition)**

```go
type DatabaseAdapter struct {
    legacyDB LegacyDatabase // Composition
}

func (d *DatabaseAdapter) Query(sql string) (*Result, error) {
    // Adapt method call
    return d.legacyDB.ExecuteQuery(sql)
}
```

2. **Class Adapter (Inheritance - simulated with embedding)**

```go
type DatabaseAdapter struct {
    LegacyDatabase // Embedding (Go's way of inheritance)
}

func (d *DatabaseAdapter) Query(sql string) (*Result, error) {
    // Directly use embedded method or adapt it
    return d.ExecuteQuery(sql)
}
```

3. **Two-Way Adapter**

```go
type TwoWayAdapter struct {
    serviceA ServiceA
    serviceB ServiceB
}

func (t *TwoWayAdapter) ConvertAToB(req ServiceARequest) ServiceBRequest {
    // Convert A format to B format
}

func (t *TwoWayAdapter) ConvertBToA(req ServiceBRequest) ServiceARequest {
    // Convert B format to A format
}
```

4. **Pluggable Adapter**

```go
type AdapterRegistry struct {
    adapters map[string]PaymentProcessor
}

func (r *AdapterRegistry) Register(name string, adapter PaymentProcessor) {
    r.adapters[name] = adapter
}

func (r *AdapterRegistry) GetAdapter(name string) (PaymentProcessor, error) {
    adapter, exists := r.adapters[name]
    if !exists {
        return nil, fmt.Errorf("adapter not found: %s", name)
    }
    return adapter, nil
}
```

### Trade-offs

**Pros:**

- **Integration**: Enables integration with incompatible interfaces
- **Reusability**: Allows reuse of existing functionality
- **Flexibility**: Easy to switch between different implementations
- **Legacy Support**: Enables gradual migration from legacy systems
- **Decoupling**: Decouples client code from specific implementations

**Cons:**

- **Complexity**: Adds an extra layer of abstraction
- **Performance**: May introduce overhead due to extra calls
- **Maintenance**: Need to maintain adapter code
- **Feature Gaps**: May not expose all features of the adaptee
- **Error Handling**: Error handling becomes more complex

**When to Choose Adapter vs Alternatives:**

| Scenario                 | Pattern   | Reason              |
| ------------------------ | --------- | ------------------- |
| Incompatible interfaces  | Adapter   | Enables integration |
| Need to modify interface | Decorator | Adds behavior       |
| Multiple implementations | Strategy  | Runtime selection   |
| Hide complex subsystem   | Facade    | Simplify interface  |
| Legacy integration       | Adapter   | Gradual migration   |

## Testable Example

```go
package main

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/stretchr/testify/mock"
    "github.com/shopspring/decimal"
)

// Mock Razorpay Gateway for testing
type MockRazorpayGateway struct {
    mock.Mock
}

func (m *MockRazorpayGateway) CreateOrder(amount int, currency, receipt string, notes map[string]interface{}) (*RazorpayOrder, error) {
    args := m.Called(amount, currency, receipt, notes)
    return args.Get(0).(*RazorpayOrder), args.Error(1)
}

func (m *MockRazorpayGateway) CapturePayment(paymentID string, amount int) (*RazorpayPayment, error) {
    args := m.Called(paymentID, amount)
    return args.Get(0).(*RazorpayPayment), args.Error(1)
}

func (m *MockRazorpayGateway) RefundPayment(paymentID string, amount int) (*RazorpayRefund, error) {
    args := m.Called(paymentID, amount)
    return args.Get(0).(*RazorpayRefund), args.Error(1)
}

func (m *MockRazorpayGateway) FetchPayment(paymentID string) (*RazorpayPayment, error) {
    args := m.Called(paymentID)
    return args.Get(0).(*RazorpayPayment), args.Error(1)
}

// Mock Stripe Gateway for testing
type MockStripeGateway struct {
    mock.Mock
}

func (m *MockStripeGateway) CreatePaymentIntent(amount int64, currency, description string, metadata map[string]string) (*StripePaymentIntent, error) {
    args := m.Called(amount, currency, description, metadata)
    return args.Get(0).(*StripePaymentIntent), args.Error(1)
}

func (m *MockStripeGateway) ConfirmPaymentIntent(paymentIntentID string) (*StripePaymentIntent, error) {
    args := m.Called(paymentIntentID)
    return args.Get(0).(*StripePaymentIntent), args.Error(1)
}

func (m *MockStripeGateway) CreateRefund(paymentIntentID string, amount int64) (*StripeRefund, error) {
    args := m.Called(paymentIntentID, amount)
    return args.Get(0).(*StripeRefund), args.Error(1)
}

func (m *MockStripeGateway) RetrievePaymentIntent(paymentIntentID string) (*StripePaymentIntent, error) {
    args := m.Called(paymentIntentID)
    return args.Get(0).(*StripePaymentIntent), args.Error(1)
}

func TestRazorpayAdapter_ProcessPayment(t *testing.T) {
    mockGateway := &MockRazorpayGateway{}
    adapter := NewRazorpayAdapter(mockGateway)

    // Setup mock expectations
    expectedOrder := &RazorpayOrder{
        ID:        "order_123",
        Entity:    "order",
        Amount:    9999, // 99.99 in paise
        Currency:  "INR",
        Status:    "created",
        CreatedAt: 1640995200,
    }

    mockGateway.On("CreateOrder", 9999, "INR", "ORDER_456", mock.AnythingOfType("map[string]interface {}")).
        Return(expectedOrder, nil)

    // Create payment request
    req := PaymentRequest{
        CustomerID:    "CUST_123",
        OrderID:       "ORDER_456",
        Amount:        decimal.NewFromFloat(99.99),
        Currency:      "INR",
        PaymentMethod: "card",
        Description:   "Test payment",
        Metadata: map[string]string{
            "test": "value",
        },
    }

    // Process payment
    response, err := adapter.ProcessPayment(req)

    // Assertions
    require.NoError(t, err)
    assert.Equal(t, "order_123", response.TransactionID)
    assert.Equal(t, "pending", response.Status)
    assert.Equal(t, decimal.NewFromFloat(99.99), response.Amount)
    assert.Equal(t, "INR", response.Currency)
    assert.Contains(t, response.PaymentURL, "order_123")

    mockGateway.AssertExpectations(t)
}

func TestStripeAdapter_ProcessPayment(t *testing.T) {
    mockGateway := &MockStripeGateway{}
    adapter := NewStripeAdapter(mockGateway)

    // Setup mock expectations
    expectedIntent := &StripePaymentIntent{
        ID:           "pi_123",
        Object:       "payment_intent",
        Amount:       9999, // 99.99 in cents
        Currency:     "usd",
        Status:       "requires_payment_method",
        ClientSecret: "pi_123_secret",
        Created:      1640995200,
    }

    mockGateway.On("CreatePaymentIntent", int64(9999), "usd", "Test payment", mock.AnythingOfType("map[string]string")).
        Return(expectedIntent, nil)

    // Create payment request
    req := PaymentRequest{
        CustomerID:    "CUST_123",
        OrderID:       "ORDER_456",
        Amount:        decimal.NewFromFloat(99.99),
        Currency:      "usd",
        PaymentMethod: "card",
        Description:   "Test payment",
        Metadata: map[string]string{
            "test": "value",
        },
    }

    // Process payment
    response, err := adapter.ProcessPayment(req)

    // Assertions
    require.NoError(t, err)
    assert.Equal(t, "pi_123", response.TransactionID)
    assert.Equal(t, "pending", response.Status)
    assert.Equal(t, decimal.NewFromFloat(99.99), response.Amount)
    assert.Equal(t, "usd", response.Currency)
    assert.Contains(t, response.PaymentURL, "pi_123_secret")

    mockGateway.AssertExpectations(t)
}

func TestRazorpayAdapter_RefundPayment(t *testing.T) {
    mockGateway := &MockRazorpayGateway{}
    adapter := NewRazorpayAdapter(mockGateway)

    // Setup mock expectations
    expectedRefund := &RazorpayRefund{
        ID:        "rfnd_123",
        Entity:    "refund",
        Amount:    2999, // 29.99 in paise
        PaymentID: "pay_123",
        Status:    "processed",
        CreatedAt: 1640995200,
    }

    mockGateway.On("RefundPayment", "pay_123", 2999).
        Return(expectedRefund, nil)

    // Process refund
    response, err := adapter.RefundPayment("pay_123", decimal.NewFromFloat(29.99))

    // Assertions
    require.NoError(t, err)
    assert.Equal(t, "rfnd_123", response.RefundID)
    assert.Equal(t, "pay_123", response.TransactionID)
    assert.Equal(t, decimal.NewFromFloat(29.99), response.Amount)
    assert.Equal(t, "processed", response.Status)

    mockGateway.AssertExpectations(t)
}

func TestStripeAdapter_RefundPayment(t *testing.T) {
    mockGateway := &MockStripeGateway{}
    adapter := NewStripeAdapter(mockGateway)

    // Setup mock expectations
    expectedRefund := &StripeRefund{
        ID:              "re_123",
        Object:          "refund",
        Amount:          2999, // 29.99 in cents
        PaymentIntentID: "pi_123",
        Status:          "succeeded",
        Created:         1640995200,
    }

    mockGateway.On("CreateRefund", "pi_123", int64(2999)).
        Return(expectedRefund, nil)

    // Process refund
    response, err := adapter.RefundPayment("pi_123", decimal.NewFromFloat(29.99))

    // Assertions
    require.NoError(t, err)
    assert.Equal(t, "re_123", response.RefundID)
    assert.Equal(t, "pi_123", response.TransactionID)
    assert.Equal(t, decimal.NewFromFloat(29.99), response.Amount)
    assert.Equal(t, "succeeded", response.Status)

    mockGateway.AssertExpectations(t)
}

func TestPaymentService_ProcessPayment(t *testing.T) {
    paymentService := NewPaymentService()

    // Create mock adapters
    mockRazorpayGateway := &MockRazorpayGateway{}
    razorpayAdapter := NewRazorpayAdapter(mockRazorpayGateway)

    mockStripeGateway := &MockStripeGateway{}
    stripeAdapter := NewStripeAdapter(mockStripeGateway)

    // Register adapters
    paymentService.RegisterProcessor("razorpay", razorpayAdapter)
    paymentService.RegisterProcessor("stripe", stripeAdapter)

    // Setup expectations
    mockRazorpayGateway.On("CreateOrder", mock.Anything, mock.Anything, mock.Anything, mock.Anything).
        Return(&RazorpayOrder{ID: "order_123", Status: "created"}, nil)

    mockStripeGateway.On("CreatePaymentIntent", mock.Anything, mock.Anything, mock.Anything, mock.Anything).
        Return(&StripePaymentIntent{ID: "pi_123", Status: "requires_payment_method"}, nil)

    req := PaymentRequest{
        Amount:   decimal.NewFromFloat(100.00),
        Currency: "USD",
        OrderID:  "ORDER_123",
    }

    // Test Razorpay
    response, err := paymentService.ProcessPayment("razorpay", req)
    assert.NoError(t, err)
    assert.Equal(t, "order_123", response.TransactionID)

    // Test Stripe
    response, err = paymentService.ProcessPayment("stripe", req)
    assert.NoError(t, err)
    assert.Equal(t, "pi_123", response.TransactionID)

    // Test unsupported provider
    _, err = paymentService.ProcessPayment("unsupported", req)
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "unsupported payment provider")

    // Test supported providers
    providers := paymentService.GetSupportedProviders()
    assert.Contains(t, providers, "razorpay")
    assert.Contains(t, providers, "stripe")
    assert.Len(t, providers, 2)
}

func TestRazorpayAdapter_StatusConversion(t *testing.T) {
    adapter := &RazorpayAdapter{}

    tests := []struct {
        razorpayStatus string
        expectedStatus string
    }{
        {"created", "pending"},
        {"attempted", "processing"},
        {"paid", "completed"},
        {"captured", "completed"},
        {"failed", "failed"},
        {"unknown", "unknown"},
    }

    for _, tt := range tests {
        t.Run(tt.razorpayStatus, func(t *testing.T) {
            result := adapter.convertRazorpayStatus(tt.razorpayStatus)
            assert.Equal(t, tt.expectedStatus, result)
        })
    }
}

func TestStripeAdapter_StatusConversion(t *testing.T) {
    adapter := &StripeAdapter{}

    tests := []struct {
        stripeStatus   string
        expectedStatus string
    }{
        {"requires_payment_method", "pending"},
        {"requires_confirmation", "pending"},
        {"processing", "processing"},
        {"succeeded", "completed"},
        {"requires_action", "action_required"},
        {"canceled", "cancelled"},
        {"unknown", "unknown"},
    }

    for _, tt := range tests {
        t.Run(tt.stripeStatus, func(t *testing.T) {
            result := adapter.convertStripeStatus(tt.stripeStatus)
            assert.Equal(t, tt.expectedStatus, result)
        })
    }
}

func TestAmountConversion(t *testing.T) {
    t.Run("Razorpay amount conversion", func(t *testing.T) {
        amount := decimal.NewFromFloat(99.99)
        amountInPaise := int(amount.Mul(decimal.NewFromInt(100)).IntPart())
        assert.Equal(t, 9999, amountInPaise)

        // Convert back
        converted := decimal.NewFromInt(int64(amountInPaise)).Div(decimal.NewFromInt(100))
        assert.Equal(t, amount, converted)
    })

    t.Run("Stripe amount conversion", func(t *testing.T) {
        amount := decimal.NewFromFloat(99.99)
        amountInCents := amount.Mul(decimal.NewFromInt(100)).IntPart()
        assert.Equal(t, int64(9999), amountInCents)

        // Convert back
        converted := decimal.NewFromInt(amountInCents).Div(decimal.NewFromInt(100))
        assert.Equal(t, amount, converted)
    })
}

func BenchmarkRazorpayAdapter_ProcessPayment(b *testing.B) {
    gateway := NewRazorpayGateway("test_key", "test_secret")
    adapter := NewRazorpayAdapter(gateway)

    req := PaymentRequest{
        CustomerID:    "CUST_123",
        OrderID:       "ORDER_456",
        Amount:        decimal.NewFromFloat(99.99),
        Currency:      "INR",
        PaymentMethod: "card",
    }

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        _, err := adapter.ProcessPayment(req)
        if err != nil {
            b.Fatal(err)
        }
    }
}

func BenchmarkStripeAdapter_ProcessPayment(b *testing.B) {
    gateway := NewStripeGateway("test_key")
    adapter := NewStripeAdapter(gateway)

    req := PaymentRequest{
        CustomerID:    "CUST_123",
        OrderID:       "ORDER_456",
        Amount:        decimal.NewFromFloat(99.99),
        Currency:      "usd",
        PaymentMethod: "card",
    }

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        _, err := adapter.ProcessPayment(req)
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

## Integration Tips

### 1. Configuration-Based Adapter Selection

```go
type AdapterConfig struct {
    Provider string                 `yaml:"provider"`
    Config   map[string]interface{} `yaml:"config"`
}

type AdapterFactory struct {
    configs map[string]AdapterConfig
}

func (f *AdapterFactory) CreateAdapter(provider string) (PaymentProcessor, error) {
    config, exists := f.configs[provider]
    if !exists {
        return nil, fmt.Errorf("unknown provider: %s", provider)
    }

    switch provider {
    case "razorpay":
        return f.createRazorpayAdapter(config.Config)
    case "stripe":
        return f.createStripeAdapter(config.Config)
    default:
        return nil, fmt.Errorf("unsupported provider: %s", provider)
    }
}
```

### 2. Health Check Integration

```go
type HealthCheckableAdapter interface {
    PaymentProcessor
    HealthCheck() error
}

type RazorpayAdapterWithHealthCheck struct {
    *RazorpayAdapter
}

func (r *RazorpayAdapterWithHealthCheck) HealthCheck() error {
    // Perform health check on Razorpay gateway
    _, err := r.gateway.CreateOrder(1, "INR", "health_check", nil)
    return err
}
```

### 3. Metrics Integration

```go
type MetricsAdapter struct {
    adapter PaymentProcessor
    metrics Metrics
}

func (m *MetricsAdapter) ProcessPayment(req PaymentRequest) (*PaymentResponse, error) {
    start := time.Now()
    response, err := m.adapter.ProcessPayment(req)
    duration := time.Since(start)

    if err != nil {
        m.metrics.IncCounter("payment_errors", map[string]string{
            "provider": m.getProviderName(),
        })
    } else {
        m.metrics.IncCounter("payment_success", map[string]string{
            "provider": m.getProviderName(),
        })
    }

    m.metrics.RecordDuration("payment_duration", duration, map[string]string{
        "provider": m.getProviderName(),
    })

    return response, err
}
```

### 4. Circuit Breaker Integration

```go
type CircuitBreakerAdapter struct {
    adapter        PaymentProcessor
    circuitBreaker *CircuitBreaker
}

func (c *CircuitBreakerAdapter) ProcessPayment(req PaymentRequest) (*PaymentResponse, error) {
    result, err := c.circuitBreaker.Execute(func() (interface{}, error) {
        return c.adapter.ProcessPayment(req)
    })

    if err != nil {
        return nil, err
    }

    return result.(*PaymentResponse), nil
}
```

## Common Interview Questions

### 1. **How does Adapter differ from Facade pattern?**

**Answer:**
| Aspect | Adapter | Facade |
|--------|---------|--------|
| **Purpose** | Make incompatible interfaces compatible | Simplify complex subsystem |
| **Interface** | Adapts existing interface to expected one | Provides simplified interface |
| **Relationship** | One-to-one adaptation | One-to-many simplification |
| **Use Case** | Integration with legacy/third-party systems | Hide complexity of subsystems |

**Example:**

```go
// Adapter - makes LegacyAPI compatible with ModernAPI
type LegacyAPIAdapter struct {
    legacyAPI *LegacyAPI
}

func (a *LegacyAPIAdapter) ModernMethod() error {
    return a.legacyAPI.LegacyMethod() // Adapts call
}

// Facade - simplifies complex subsystem
type PaymentFacade struct {
    validator    *PaymentValidator
    processor    *PaymentProcessor
    logger       *PaymentLogger
    notification *NotificationService
}

func (f *PaymentFacade) ProcessPayment(req PaymentRequest) error {
    // Coordinates multiple subsystems
    f.validator.Validate(req)
    f.processor.Process(req)
    f.logger.Log(req)
    f.notification.Send(req)
    return nil
}
```

### 2. **When would you use Object Adapter vs Class Adapter?**

**Answer:**
In Go, we primarily use Object Adapter (composition) since Go doesn't have inheritance. However, we can simulate Class Adapter using embedding:

**Object Adapter (Composition):**

```go
type PaymentAdapter struct {
    gateway PaymentGateway // Composition
}

func (p *PaymentAdapter) ProcessPayment(req Request) error {
    // Delegate to gateway
    return p.gateway.Process(req)
}
```

**Pros:**

- More flexible (can adapt multiple objects)
- Can adapt entire hierarchy
- Runtime adapter selection

**Class Adapter (Embedding in Go):**

```go
type PaymentAdapter struct {
    PaymentGateway // Embedding
}

func (p *PaymentAdapter) ProcessPayment(req Request) error {
    // Can directly call embedded methods
    return p.Process(req) // or adapt the call
}
```

**Pros:**

- Direct access to adaptee methods
- Slightly better performance
- Smaller object footprint

**Cons:**

- Less flexible
- Can only adapt one class

### 3. **How do you handle multiple incompatible interfaces that need to be adapted?**

**Answer:**
Use a combination of patterns and strategies:

1. **Chain of Adapters:**

```go
type ChainedAdapter struct {
    adapters []Adapter
}

func (c *ChainedAdapter) Adapt(input interface{}) (interface{}, error) {
    result := input
    for _, adapter := range c.adapters {
        var err error
        result, err = adapter.Adapt(result)
        if err != nil {
            return nil, err
        }
    }
    return result, nil
}
```

2. **Composite Adapter:**

```go
type CompositeAdapter struct {
    primaryAdapter   PaymentProcessor
    secondaryAdapter PaymentProcessor
    fallbackAdapter  PaymentProcessor
}

func (c *CompositeAdapter) ProcessPayment(req PaymentRequest) (*PaymentResponse, error) {
    // Try primary first
    if resp, err := c.primaryAdapter.ProcessPayment(req); err == nil {
        return resp, nil
    }

    // Try secondary
    if resp, err := c.secondaryAdapter.ProcessPayment(req); err == nil {
        return resp, nil
    }

    // Fallback
    return c.fallbackAdapter.ProcessPayment(req)
}
```

3. **Registry-Based Adapter:**

```go
type AdapterRegistry struct {
    adapters map[string]Adapter
}

func (r *AdapterRegistry) GetAdapter(protocol string) (Adapter, error) {
    adapter, exists := r.adapters[protocol]
    if !exists {
        return nil, fmt.Errorf("no adapter for protocol: %s", protocol)
    }
    return adapter, nil
}
```

### 4. **How do you test Adapter implementations effectively?**

**Answer:**
Testing adapters requires focusing on the adaptation logic:

1. **Mock the Adaptee:**

```go
func TestPaymentAdapter(t *testing.T) {
    mockGateway := &MockPaymentGateway{}
    adapter := NewPaymentAdapter(mockGateway)

    // Test successful adaptation
    mockGateway.On("CreateOrder", mock.Anything).Return(&Order{}, nil)

    req := PaymentRequest{Amount: decimal.NewFromFloat(100.0)}
    resp, err := adapter.ProcessPayment(req)

    assert.NoError(t, err)
    assert.NotNil(t, resp)
    mockGateway.AssertExpectations(t)
}
```

2. **Test Data Conversion:**

```go
func TestAmountConversion(t *testing.T) {
    tests := []struct {
        input    decimal.Decimal
        expected int
    }{
        {decimal.NewFromFloat(99.99), 9999},
        {decimal.NewFromFloat(100.00), 10000},
        {decimal.NewFromFloat(0.01), 1},
    }

    for _, tt := range tests {
        result := convertToSmallestUnit(tt.input)
        assert.Equal(t, tt.expected, result)
    }
}
```

3. **Test Error Handling:**

```go
func TestAdapterErrorHandling(t *testing.T) {
    mockGateway := &MockPaymentGateway{}
    adapter := NewPaymentAdapter(mockGateway)

    mockGateway.On("CreateOrder", mock.Anything).
        Return(nil, errors.New("gateway error"))

    req := PaymentRequest{Amount: decimal.NewFromFloat(100.0)}
    _, err := adapter.ProcessPayment(req)

    assert.Error(t, err)
    assert.Contains(t, err.Error(), "gateway error")
}
```

4. **Integration Tests:**

```go
func TestAdapterIntegration(t *testing.T) {
    // Use real gateway in test environment
    gateway := NewRealPaymentGateway(testConfig)
    adapter := NewPaymentAdapter(gateway)

    req := PaymentRequest{
        Amount:   decimal.NewFromFloat(1.00), // Small test amount
        Currency: "USD",
    }

    resp, err := adapter.ProcessPayment(req)

    assert.NoError(t, err)
    assert.NotEmpty(t, resp.TransactionID)
}
```

### 5. **How do you handle versioning and backward compatibility with Adapters?**

**Answer:**
Use versioned adapters and compatibility layers:

1. **Versioned Adapters:**

```go
type PaymentAdapterV1 struct {
    gateway PaymentGatewayV1
}

type PaymentAdapterV2 struct {
    gateway PaymentGatewayV2
}

type VersionedAdapterFactory struct {
    v1Adapters map[string]*PaymentAdapterV1
    v2Adapters map[string]*PaymentAdapterV2
}

func (f *VersionedAdapterFactory) GetAdapter(provider, version string) (PaymentProcessor, error) {
    switch version {
    case "v1":
        return f.v1Adapters[provider], nil
    case "v2":
        return f.v2Adapters[provider], nil
    default:
        return nil, fmt.Errorf("unsupported version: %s", version)
    }
}
```

2. **Backward Compatibility Layer:**

```go
type BackwardCompatibleAdapter struct {
    newAdapter PaymentProcessorV2
}

func (b *BackwardCompatibleAdapter) ProcessPayment(req PaymentRequestV1) (*PaymentResponseV1, error) {
    // Convert V1 request to V2
    v2Req := b.convertV1ToV2(req)

    // Process with V2 adapter
    v2Resp, err := b.newAdapter.ProcessPayment(v2Req)
    if err != nil {
        return nil, err
    }

    // Convert V2 response back to V1
    return b.convertV2ToV1(v2Resp), nil
}
```

3. **Migration Strategy:**

```go
type MigrationAdapter struct {
    legacyAdapter PaymentProcessor
    newAdapter    PaymentProcessor
    migrationFlag bool
}

func (m *MigrationAdapter) ProcessPayment(req PaymentRequest) (*PaymentResponse, error) {
    if m.migrationFlag {
        // Try new adapter first, fallback to legacy
        if resp, err := m.newAdapter.ProcessPayment(req); err == nil {
            return resp, nil
        }
        log.Println("New adapter failed, falling back to legacy")
        return m.legacyAdapter.ProcessPayment(req)
    }

    // Use legacy adapter
    return m.legacyAdapter.ProcessPayment(req)
}
```
