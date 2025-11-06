---
# Auto-generated front matter
Title: Builder
LastUpdated: 2025-11-06T20:45:58.512846
Tags: []
Status: draft
---

# Builder Pattern

## Pattern Name & Intent

**Builder** is a creational design pattern that lets you construct complex objects step by step. The pattern allows you to produce different types and representations of an object using the same construction code.

**Key Intent:**

- Separate the construction of complex objects from their representation
- Allow step-by-step construction with optional parameters
- Create different representations of the same object
- Provide a fluent interface for object construction
- Handle optional parameters elegantly without telescoping constructors

## When to Use

**Use Builder when:**

1. **Complex Construction**: Object has many parameters, especially optional ones
2. **Telescoping Constructor Problem**: You want to avoid constructors with many parameters
3. **Immutable Objects**: You need to create immutable objects with validation
4. **Multiple Representations**: Same construction process should create different representations
5. **Step-by-Step Construction**: Object needs to be built in multiple steps
6. **Validation During Construction**: Need to validate object state during construction

**Don't use when:**

- Object has few parameters (< 4-5)
- Construction is simple and straightforward
- All parameters are required
- Object is mutable and can be modified after creation

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Request Builder

```go
// Complex payment request with many optional parameters
type PaymentRequest struct {
    Amount      float64
    Currency    string
    OrderID     string
    UserID      string
    Description string
    Metadata    map[string]string
    Webhook     *WebhookConfig
    Retry       *RetryConfig
    Timeout     time.Duration
    Notes       []string
}

// Builder allows optional configuration
payment := NewPaymentRequestBuilder().
    Amount(100.50, "USD").
    Order("ORD_123").
    User("USER_456").
    Description("Product purchase").
    WithWebhook("https://api.example.com/webhook").
    WithRetry(3, time.Minute*5).
    AddNote("VIP customer").
    Build()
```

### 2. Financial Report Builder

```go
// Complex financial report with multiple sections
type FinancialReport struct {
    Title       string
    Period      DateRange
    Sections    []ReportSection
    Charts      []Chart
    Filters     []Filter
    Format      ReportFormat
    Recipients  []string
    Schedule    *ScheduleConfig
}

// Builder for step-by-step report construction
report := NewFinancialReportBuilder().
    Title("Monthly Revenue Report").
    Period(lastMonth).
    AddRevenueSection().
    AddExpenseSection().
    AddPieChart("Revenue by Category").
    FilterByRegion("North America").
    Format(PDF).
    AddRecipient("cfo@company.com").
    ScheduleMonthly().
    Build()
```

### 3. Trading Strategy Builder

```go
// Complex trading strategy configuration
type TradingStrategy struct {
    Name        string
    Instruments []string
    Indicators  []TechnicalIndicator
    Rules       []TradingRule
    RiskMgmt    RiskManagement
    Backtest    BacktestConfig
    Execution   ExecutionConfig
}

// Builder for strategy construction
strategy := NewTradingStrategyBuilder().
    Name("Mean Reversion Strategy").
    AddInstruments("AAPL", "GOOGL", "MSFT").
    AddRSIIndicator(14, 30, 70).
    AddMovingAverage(20, "SMA").
    AddRule("BUY when RSI < 30").
    SetRiskLimit(0.02). // 2% risk per trade
    EnableBacktest(oneYearAgo, today).
    SetExecutionMode("PAPER").
    Build()
```

### 4. Bank Account Builder

```go
// Complex bank account with various features
type BankAccount struct {
    AccountNumber string
    AccountType   AccountType
    Currency      string
    Owner         CustomerInfo
    Features      []AccountFeature
    Limits        AccountLimits
    Notifications []NotificationSetting
    Cards         []DebitCard
}

// Builder for account setup
account := NewBankAccountBuilder().
    Type(CHECKING).
    Currency("USD").
    Owner(customer).
    EnableOnlineBanking().
    EnableMobileBanking().
    SetDailyLimit(5000).
    SetMonthlyLimit(50000).
    AddSMSNotifications().
    AddEmailNotifications().
    IssueDebitCard().
    Build()
```

## Go Implementation

```go
package main

import (
    "fmt"
    "time"
    "errors"
    "strings"
    "net/url"
)

// Product - Complex Payment Request
type PaymentRequest struct {
    // Required fields
    amount   float64
    currency string
    orderID  string
    userID   string

    // Optional fields
    description string
    metadata    map[string]string
    webhook     *WebhookConfig
    retry       *RetryConfig
    timeout     time.Duration
    notes       []string
    tags        []string

    // Internal fields
    createdAt   time.Time
    requestID   string
    validated   bool
}

type WebhookConfig struct {
    URL     string
    Headers map[string]string
    Secret  string
}

type RetryConfig struct {
    MaxAttempts int
    Interval    time.Duration
    Backoff     float64
}

// Validation methods
func (p *PaymentRequest) Validate() error {
    if p.amount <= 0 {
        return errors.New("amount must be positive")
    }

    if p.currency == "" {
        return errors.New("currency is required")
    }

    if p.orderID == "" {
        return errors.New("order ID is required")
    }

    if p.userID == "" {
        return errors.New("user ID is required")
    }

    if p.webhook != nil {
        if _, err := url.Parse(p.webhook.URL); err != nil {
            return fmt.Errorf("invalid webhook URL: %w", err)
        }
    }

    if p.timeout < 0 {
        return errors.New("timeout cannot be negative")
    }

    return nil
}

func (p *PaymentRequest) String() string {
    var parts []string
    parts = append(parts, fmt.Sprintf("Amount: %.2f %s", p.amount, p.currency))
    parts = append(parts, fmt.Sprintf("Order: %s", p.orderID))
    parts = append(parts, fmt.Sprintf("User: %s", p.userID))

    if p.description != "" {
        parts = append(parts, fmt.Sprintf("Description: %s", p.description))
    }

    if len(p.notes) > 0 {
        parts = append(parts, fmt.Sprintf("Notes: %v", p.notes))
    }

    return fmt.Sprintf("PaymentRequest{%s}", strings.Join(parts, ", "))
}

// Getters (since fields are private)
func (p *PaymentRequest) GetAmount() float64 { return p.amount }
func (p *PaymentRequest) GetCurrency() string { return p.currency }
func (p *PaymentRequest) GetOrderID() string { return p.orderID }
func (p *PaymentRequest) GetUserID() string { return p.userID }
func (p *PaymentRequest) GetDescription() string { return p.description }
func (p *PaymentRequest) GetMetadata() map[string]string { return p.metadata }
func (p *PaymentRequest) GetWebhook() *WebhookConfig { return p.webhook }
func (p *PaymentRequest) GetRetry() *RetryConfig { return p.retry }
func (p *PaymentRequest) GetTimeout() time.Duration { return p.timeout }
func (p *PaymentRequest) GetNotes() []string { return p.notes }
func (p *PaymentRequest) GetTags() []string { return p.tags }
func (p *PaymentRequest) GetCreatedAt() time.Time { return p.createdAt }
func (p *PaymentRequest) GetRequestID() string { return p.requestID }
func (p *PaymentRequest) IsValidated() bool { return p.validated }

// Builder Interface
type PaymentRequestBuilder interface {
    // Required fields
    Amount(amount float64, currency string) PaymentRequestBuilder
    Order(orderID string) PaymentRequestBuilder
    User(userID string) PaymentRequestBuilder

    // Optional fields
    Description(description string) PaymentRequestBuilder
    AddMetadata(key, value string) PaymentRequestBuilder
    SetMetadata(metadata map[string]string) PaymentRequestBuilder
    WithWebhook(url string) PaymentRequestBuilder
    WithWebhookHeaders(headers map[string]string) PaymentRequestBuilder
    WithWebhookSecret(secret string) PaymentRequestBuilder
    WithRetry(maxAttempts int, interval time.Duration) PaymentRequestBuilder
    WithRetryBackoff(backoff float64) PaymentRequestBuilder
    WithTimeout(timeout time.Duration) PaymentRequestBuilder
    AddNote(note string) PaymentRequestBuilder
    AddNotes(notes ...string) PaymentRequestBuilder
    AddTag(tag string) PaymentRequestBuilder
    AddTags(tags ...string) PaymentRequestBuilder

    // Build and validation
    Build() (*PaymentRequest, error)
    BuildUnsafe() *PaymentRequest
    Reset() PaymentRequestBuilder
    Clone() PaymentRequestBuilder
}

// Concrete Builder
type paymentRequestBuilder struct {
    request *PaymentRequest
    errors  []error
}

// Constructor
func NewPaymentRequestBuilder() PaymentRequestBuilder {
    return &paymentRequestBuilder{
        request: &PaymentRequest{
            metadata:  make(map[string]string),
            notes:     make([]string, 0),
            tags:      make([]string, 0),
            createdAt: time.Now(),
            requestID: generateRequestID(),
        },
        errors: make([]error, 0),
    }
}

// Required field methods
func (b *paymentRequestBuilder) Amount(amount float64, currency string) PaymentRequestBuilder {
    if amount <= 0 {
        b.errors = append(b.errors, errors.New("amount must be positive"))
    }
    if currency == "" {
        b.errors = append(b.errors, errors.New("currency cannot be empty"))
    }

    b.request.amount = amount
    b.request.currency = strings.ToUpper(currency)
    return b
}

func (b *paymentRequestBuilder) Order(orderID string) PaymentRequestBuilder {
    if orderID == "" {
        b.errors = append(b.errors, errors.New("order ID cannot be empty"))
    }

    b.request.orderID = orderID
    return b
}

func (b *paymentRequestBuilder) User(userID string) PaymentRequestBuilder {
    if userID == "" {
        b.errors = append(b.errors, errors.New("user ID cannot be empty"))
    }

    b.request.userID = userID
    return b
}

// Optional field methods
func (b *paymentRequestBuilder) Description(description string) PaymentRequestBuilder {
    b.request.description = description
    return b
}

func (b *paymentRequestBuilder) AddMetadata(key, value string) PaymentRequestBuilder {
    if key == "" {
        b.errors = append(b.errors, errors.New("metadata key cannot be empty"))
        return b
    }

    b.request.metadata[key] = value
    return b
}

func (b *paymentRequestBuilder) SetMetadata(metadata map[string]string) PaymentRequestBuilder {
    if metadata != nil {
        b.request.metadata = make(map[string]string)
        for k, v := range metadata {
            b.request.metadata[k] = v
        }
    }
    return b
}

func (b *paymentRequestBuilder) WithWebhook(webhookURL string) PaymentRequestBuilder {
    if webhookURL == "" {
        b.errors = append(b.errors, errors.New("webhook URL cannot be empty"))
        return b
    }

    if _, err := url.Parse(webhookURL); err != nil {
        b.errors = append(b.errors, fmt.Errorf("invalid webhook URL: %w", err))
        return b
    }

    if b.request.webhook == nil {
        b.request.webhook = &WebhookConfig{
            Headers: make(map[string]string),
        }
    }
    b.request.webhook.URL = webhookURL
    return b
}

func (b *paymentRequestBuilder) WithWebhookHeaders(headers map[string]string) PaymentRequestBuilder {
    if b.request.webhook == nil {
        b.request.webhook = &WebhookConfig{
            Headers: make(map[string]string),
        }
    }

    for k, v := range headers {
        b.request.webhook.Headers[k] = v
    }
    return b
}

func (b *paymentRequestBuilder) WithWebhookSecret(secret string) PaymentRequestBuilder {
    if b.request.webhook == nil {
        b.request.webhook = &WebhookConfig{
            Headers: make(map[string]string),
        }
    }

    b.request.webhook.Secret = secret
    return b
}

func (b *paymentRequestBuilder) WithRetry(maxAttempts int, interval time.Duration) PaymentRequestBuilder {
    if maxAttempts < 0 {
        b.errors = append(b.errors, errors.New("max attempts cannot be negative"))
    }

    if interval < 0 {
        b.errors = append(b.errors, errors.New("retry interval cannot be negative"))
    }

    b.request.retry = &RetryConfig{
        MaxAttempts: maxAttempts,
        Interval:    interval,
        Backoff:     1.0, // default backoff
    }
    return b
}

func (b *paymentRequestBuilder) WithRetryBackoff(backoff float64) PaymentRequestBuilder {
    if backoff <= 0 {
        b.errors = append(b.errors, errors.New("retry backoff must be positive"))
        return b
    }

    if b.request.retry == nil {
        b.request.retry = &RetryConfig{
            MaxAttempts: 3,
            Interval:    time.Second * 30,
        }
    }

    b.request.retry.Backoff = backoff
    return b
}

func (b *paymentRequestBuilder) WithTimeout(timeout time.Duration) PaymentRequestBuilder {
    if timeout < 0 {
        b.errors = append(b.errors, errors.New("timeout cannot be negative"))
    }

    b.request.timeout = timeout
    return b
}

func (b *paymentRequestBuilder) AddNote(note string) PaymentRequestBuilder {
    if note != "" {
        b.request.notes = append(b.request.notes, note)
    }
    return b
}

func (b *paymentRequestBuilder) AddNotes(notes ...string) PaymentRequestBuilder {
    for _, note := range notes {
        if note != "" {
            b.request.notes = append(b.request.notes, note)
        }
    }
    return b
}

func (b *paymentRequestBuilder) AddTag(tag string) PaymentRequestBuilder {
    if tag != "" {
        b.request.tags = append(b.request.tags, tag)
    }
    return b
}

func (b *paymentRequestBuilder) AddTags(tags ...string) PaymentRequestBuilder {
    for _, tag := range tags {
        if tag != "" {
            b.request.tags = append(b.request.tags, tag)
        }
    }
    return b
}

// Build methods
func (b *paymentRequestBuilder) Build() (*PaymentRequest, error) {
    // Check for builder errors
    if len(b.errors) > 0 {
        return nil, fmt.Errorf("builder errors: %v", b.errors)
    }

    // Validate the built object
    if err := b.request.Validate(); err != nil {
        return nil, fmt.Errorf("validation failed: %w", err)
    }

    // Mark as validated
    b.request.validated = true

    // Return a copy to ensure immutability
    return b.copyRequest(), nil
}

func (b *paymentRequestBuilder) BuildUnsafe() *PaymentRequest {
    return b.copyRequest()
}

func (b *paymentRequestBuilder) Reset() PaymentRequestBuilder {
    b.request = &PaymentRequest{
        metadata:  make(map[string]string),
        notes:     make([]string, 0),
        tags:      make([]string, 0),
        createdAt: time.Now(),
        requestID: generateRequestID(),
    }
    b.errors = make([]error, 0)
    return b
}

func (b *paymentRequestBuilder) Clone() PaymentRequestBuilder {
    newBuilder := &paymentRequestBuilder{
        request: b.copyRequest(),
        errors:  make([]error, len(b.errors)),
    }
    copy(newBuilder.errors, b.errors)
    return newBuilder
}

// Helper methods
func (b *paymentRequestBuilder) copyRequest() *PaymentRequest {
    copy := &PaymentRequest{
        amount:      b.request.amount,
        currency:    b.request.currency,
        orderID:     b.request.orderID,
        userID:      b.request.userID,
        description: b.request.description,
        timeout:     b.request.timeout,
        createdAt:   b.request.createdAt,
        requestID:   b.request.requestID,
        validated:   b.request.validated,
    }

    // Deep copy metadata
    copy.metadata = make(map[string]string)
    for k, v := range b.request.metadata {
        copy.metadata[k] = v
    }

    // Deep copy notes
    copy.notes = make([]string, len(b.request.notes))
    copySlice := copy.notes
    copy(copySlice, b.request.notes)

    // Deep copy tags
    copy.tags = make([]string, len(b.request.tags))
    copySlice = copy.tags
    copy(copySlice, b.request.tags)

    // Deep copy webhook
    if b.request.webhook != nil {
        copy.webhook = &WebhookConfig{
            URL:    b.request.webhook.URL,
            Secret: b.request.webhook.Secret,
            Headers: make(map[string]string),
        }
        for k, v := range b.request.webhook.Headers {
            copy.webhook.Headers[k] = v
        }
    }

    // Deep copy retry
    if b.request.retry != nil {
        copy.retry = &RetryConfig{
            MaxAttempts: b.request.retry.MaxAttempts,
            Interval:    b.request.retry.Interval,
            Backoff:     b.request.retry.Backoff,
        }
    }

    return copy
}

func generateRequestID() string {
    return fmt.Sprintf("REQ_%d", time.Now().UnixNano())
}

// Director - optional component that knows how to build specific objects
type PaymentRequestDirector struct {
    builder PaymentRequestBuilder
}

func NewPaymentRequestDirector(builder PaymentRequestBuilder) *PaymentRequestDirector {
    return &PaymentRequestDirector{builder: builder}
}

func (d *PaymentRequestDirector) BuildSimplePayment(amount float64, currency, orderID, userID string) (*PaymentRequest, error) {
    return d.builder.
        Reset().
        Amount(amount, currency).
        Order(orderID).
        User(userID).
        Build()
}

func (d *PaymentRequestDirector) BuildEcommercePayment(amount float64, currency, orderID, userID, description string) (*PaymentRequest, error) {
    return d.builder.
        Reset().
        Amount(amount, currency).
        Order(orderID).
        User(userID).
        Description(description).
        WithTimeout(time.Minute * 5).
        AddTag("ecommerce").
        Build()
}

func (d *PaymentRequestDirector) BuildSubscriptionPayment(amount float64, currency, orderID, userID string, webhookURL string) (*PaymentRequest, error) {
    return d.builder.
        Reset().
        Amount(amount, currency).
        Order(orderID).
        User(userID).
        Description("Subscription payment").
        WithWebhook(webhookURL).
        WithRetry(3, time.Minute*2).
        WithTimeout(time.Minute * 10).
        AddTag("subscription").
        AddTag("recurring").
        AddNote("Auto-generated subscription payment").
        Build()
}

func (d *PaymentRequestDirector) BuildHighValuePayment(amount float64, currency, orderID, userID string, approverID string) (*PaymentRequest, error) {
    return d.builder.
        Reset().
        Amount(amount, currency).
        Order(orderID).
        User(userID).
        Description("High-value transaction").
        AddMetadata("approver_id", approverID).
        AddMetadata("risk_level", "high").
        WithTimeout(time.Hour).
        AddTag("high-value").
        AddTag("manual-review").
        AddNote("Requires manual approval").
        AddNote(fmt.Sprintf("Approved by: %s", approverID)).
        Build()
}

// Example usage
func main() {
    fmt.Println("=== Builder Pattern Examples ===\n")

    // Example 1: Simple payment using builder directly
    fmt.Println("1. Simple Payment:")
    payment1, err := NewPaymentRequestBuilder().
        Amount(99.99, "USD").
        Order("ORD_001").
        User("USER_123").
        Description("Product purchase").
        Build()

    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("%s\n", payment1)
    }

    fmt.Println()

    // Example 2: Complex payment with all options
    fmt.Println("2. Complex Payment:")
    payment2, err := NewPaymentRequestBuilder().
        Amount(1500.50, "EUR").
        Order("ORD_002").
        User("USER_456").
        Description("Premium service subscription").
        AddMetadata("plan", "premium").
        AddMetadata("billing_cycle", "monthly").
        WithWebhook("https://api.myapp.com/webhooks/payment").
        WithWebhookHeaders(map[string]string{
            "Authorization": "Bearer token123",
            "Content-Type": "application/json",
        }).
        WithWebhookSecret("webhook_secret_123").
        WithRetry(3, time.Minute*2).
        WithRetryBackoff(1.5).
        WithTimeout(time.Minute * 10).
        AddNote("VIP customer").
        AddNote("Priority processing").
        AddTags("subscription", "premium", "vip").
        Build()

    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("%s\n", payment2)
        fmt.Printf("Webhook URL: %s\n", payment2.GetWebhook().URL)
        fmt.Printf("Retry Config: %+v\n", payment2.GetRetry())
        fmt.Printf("Metadata: %+v\n", payment2.GetMetadata())
    }

    fmt.Println()

    // Example 3: Using Director for predefined patterns
    fmt.Println("3. Using Director:")
    director := NewPaymentRequestDirector(NewPaymentRequestBuilder())

    // Simple payment
    simplePayment, err := director.BuildSimplePayment(50.00, "USD", "ORD_003", "USER_789")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Simple: %s\n", simplePayment)
    }

    // E-commerce payment
    ecommercePayment, err := director.BuildEcommercePayment(299.99, "USD", "ORD_004", "USER_789", "Electronics purchase")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("E-commerce: %s\n", ecommercePayment)
    }

    // Subscription payment
    subscriptionPayment, err := director.BuildSubscriptionPayment(29.99, "USD", "ORD_005", "USER_789", "https://api.myapp.com/webhook")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
    } else {
        fmt.Printf("Subscription: %s\n", subscriptionPayment)
    }

    fmt.Println()

    // Example 4: Error handling
    fmt.Println("4. Error Handling:")
    _, err = NewPaymentRequestBuilder().
        Amount(-100, "USD"). // Invalid amount
        Order("").           // Empty order ID
        User("USER_123").
        Build()

    if err != nil {
        fmt.Printf("Expected error: %v\n", err)
    }

    fmt.Println()

    // Example 5: Builder reuse and cloning
    fmt.Println("5. Builder Reuse and Cloning:")
    baseBuilder := NewPaymentRequestBuilder().
        Amount(100, "USD").
        User("USER_123").
        AddMetadata("source", "mobile_app")

    // Clone for order 1
    payment5a, _ := baseBuilder.Clone().
        Order("ORD_006").
        Description("Mobile purchase 1").
        Build()

    // Clone for order 2
    payment5b, _ := baseBuilder.Clone().
        Order("ORD_007").
        Description("Mobile purchase 2").
        AddTag("promotional").
        Build()

    fmt.Printf("Payment A: %s\n", payment5a)
    fmt.Printf("Payment B: %s\n", payment5b)

    fmt.Println()

    // Example 6: Unsafe building (for performance-critical paths)
    fmt.Println("6. Unsafe Building:")
    unsafePayment := NewPaymentRequestBuilder().
        Amount(25.99, "USD").
        Order("ORD_008").
        User("USER_123").
        BuildUnsafe() // No validation

    fmt.Printf("Unsafe payment: %s\n", unsafePayment)
    fmt.Printf("Is validated: %t\n", unsafePayment.IsValidated())
}
```

## Variants & Trade-offs

### Variants

1. **Fluent Builder (Method Chaining)**

```go
payment := NewPaymentRequestBuilder().
    Amount(100, "USD").
    Order("ORD_123").
    User("USER_456").
    Build()
```

2. **Step Builder (Enforced Order)**

```go
type AmountStep interface {
    Amount(float64, string) OrderStep
}

type OrderStep interface {
    Order(string) UserStep
}

type UserStep interface {
    User(string) OptionalStep
}

type OptionalStep interface {
    Description(string) OptionalStep
    Build() (*PaymentRequest, error)
}
```

3. **Functional Builder**

```go
type BuildOption func(*PaymentRequest)

func WithDescription(desc string) BuildOption {
    return func(p *PaymentRequest) {
        p.description = desc
    }
}

func NewPaymentRequest(amount float64, currency, orderID, userID string, options ...BuildOption) *PaymentRequest {
    p := &PaymentRequest{
        amount:   amount,
        currency: currency,
        orderID:  orderID,
        userID:   userID,
    }

    for _, option := range options {
        option(p)
    }

    return p
}
```

4. **Generic Builder**

```go
type Builder[T any] interface {
    Build() (T, error)
    Reset() Builder[T]
}

type PaymentRequestBuilder struct {
    Builder[*PaymentRequest]
    // implementation
}
```

### Trade-offs

**Pros:**

- **Readable Code**: Fluent interface makes code self-documenting
- **Optional Parameters**: Elegant handling of optional parameters
- **Immutability**: Can create immutable objects safely
- **Validation**: Can validate during construction
- **Flexibility**: Easy to add new optional parameters
- **Reusability**: Builders can be reused and cloned

**Cons:**

- **Verbosity**: More code than simple constructors
- **Memory Overhead**: Additional objects during construction
- **Complexity**: More complex than simple factory methods
- **Learning Curve**: Developers need to understand the pattern
- **Performance**: Slight performance overhead from method chaining

**When to Choose Builder vs Alternatives:**

| Pattern          | Use When                 | Avoid When                |
| ---------------- | ------------------------ | ------------------------- |
| Builder          | Many optional parameters | Few required parameters   |
| Constructor      | Simple objects           | Complex configuration     |
| Factory Method   | Object creation varies   | Step-by-step construction |
| Abstract Factory | Family of objects        | Single complex object     |

## Testable Example

```go
package main

import (
    "testing"
    "time"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestPaymentRequestBuilder_Basic(t *testing.T) {
    payment, err := NewPaymentRequestBuilder().
        Amount(100.50, "USD").
        Order("ORD_123").
        User("USER_456").
        Build()

    require.NoError(t, err)
    assert.Equal(t, 100.50, payment.GetAmount())
    assert.Equal(t, "USD", payment.GetCurrency())
    assert.Equal(t, "ORD_123", payment.GetOrderID())
    assert.Equal(t, "USER_456", payment.GetUserID())
    assert.True(t, payment.IsValidated())
}

func TestPaymentRequestBuilder_WithOptionalFields(t *testing.T) {
    payment, err := NewPaymentRequestBuilder().
        Amount(200.00, "EUR").
        Order("ORD_456").
        User("USER_789").
        Description("Test payment").
        AddMetadata("key1", "value1").
        AddMetadata("key2", "value2").
        WithTimeout(time.Minute * 5).
        AddNote("Test note").
        AddTag("test").
        Build()

    require.NoError(t, err)
    assert.Equal(t, "Test payment", payment.GetDescription())
    assert.Equal(t, "value1", payment.GetMetadata()["key1"])
    assert.Equal(t, "value2", payment.GetMetadata()["key2"])
    assert.Equal(t, time.Minute*5, payment.GetTimeout())
    assert.Contains(t, payment.GetNotes(), "Test note")
    assert.Contains(t, payment.GetTags(), "test")
}

func TestPaymentRequestBuilder_WithWebhook(t *testing.T) {
    payment, err := NewPaymentRequestBuilder().
        Amount(150.00, "USD").
        Order("ORD_789").
        User("USER_123").
        WithWebhook("https://api.example.com/webhook").
        WithWebhookHeaders(map[string]string{
            "Authorization": "Bearer token",
        }).
        WithWebhookSecret("secret123").
        Build()

    require.NoError(t, err)
    webhook := payment.GetWebhook()
    require.NotNil(t, webhook)
    assert.Equal(t, "https://api.example.com/webhook", webhook.URL)
    assert.Equal(t, "Bearer token", webhook.Headers["Authorization"])
    assert.Equal(t, "secret123", webhook.Secret)
}

func TestPaymentRequestBuilder_WithRetry(t *testing.T) {
    payment, err := NewPaymentRequestBuilder().
        Amount(75.00, "USD").
        Order("ORD_999").
        User("USER_456").
        WithRetry(3, time.Minute*2).
        WithRetryBackoff(1.5).
        Build()

    require.NoError(t, err)
    retry := payment.GetRetry()
    require.NotNil(t, retry)
    assert.Equal(t, 3, retry.MaxAttempts)
    assert.Equal(t, time.Minute*2, retry.Interval)
    assert.Equal(t, 1.5, retry.Backoff)
}

func TestPaymentRequestBuilder_ValidationErrors(t *testing.T) {
    tests := []struct {
        name    string
        builder func() PaymentRequestBuilder
        wantErr string
    }{
        {
            name: "negative amount",
            builder: func() PaymentRequestBuilder {
                return NewPaymentRequestBuilder().
                    Amount(-100, "USD").
                    Order("ORD_123").
                    User("USER_456")
            },
            wantErr: "amount must be positive",
        },
        {
            name: "empty currency",
            builder: func() PaymentRequestBuilder {
                return NewPaymentRequestBuilder().
                    Amount(100, "").
                    Order("ORD_123").
                    User("USER_456")
            },
            wantErr: "currency cannot be empty",
        },
        {
            name: "empty order ID",
            builder: func() PaymentRequestBuilder {
                return NewPaymentRequestBuilder().
                    Amount(100, "USD").
                    Order("").
                    User("USER_456")
            },
            wantErr: "order ID cannot be empty",
        },
        {
            name: "empty user ID",
            builder: func() PaymentRequestBuilder {
                return NewPaymentRequestBuilder().
                    Amount(100, "USD").
                    Order("ORD_123").
                    User("")
            },
            wantErr: "user ID cannot be empty",
        },
        {
            name: "invalid webhook URL",
            builder: func() PaymentRequestBuilder {
                return NewPaymentRequestBuilder().
                    Amount(100, "USD").
                    Order("ORD_123").
                    User("USER_456").
                    WithWebhook("invalid-url")
            },
            wantErr: "invalid webhook URL",
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            _, err := tt.builder().Build()
            require.Error(t, err)
            assert.Contains(t, err.Error(), tt.wantErr)
        })
    }
}

func TestPaymentRequestBuilder_Clone(t *testing.T) {
    baseBuilder := NewPaymentRequestBuilder().
        Amount(100, "USD").
        User("USER_123").
        AddMetadata("source", "test")

    // Build from cloned builder
    payment1, err := baseBuilder.Clone().
        Order("ORD_001").
        Description("Payment 1").
        Build()
    require.NoError(t, err)

    payment2, err := baseBuilder.Clone().
        Order("ORD_002").
        Description("Payment 2").
        Build()
    require.NoError(t, err)

    // Verify independence
    assert.Equal(t, "ORD_001", payment1.GetOrderID())
    assert.Equal(t, "Payment 1", payment1.GetDescription())
    assert.Equal(t, "ORD_002", payment2.GetOrderID())
    assert.Equal(t, "Payment 2", payment2.GetDescription())

    // Verify shared fields
    assert.Equal(t, "USER_123", payment1.GetUserID())
    assert.Equal(t, "USER_123", payment2.GetUserID())
    assert.Equal(t, "test", payment1.GetMetadata()["source"])
    assert.Equal(t, "test", payment2.GetMetadata()["source"])
}

func TestPaymentRequestBuilder_Reset(t *testing.T) {
    builder := NewPaymentRequestBuilder().
        Amount(100, "USD").
        Order("ORD_123").
        User("USER_456").
        Description("Test")

    // Build first payment
    payment1, err := builder.Build()
    require.NoError(t, err)
    assert.Equal(t, "Test", payment1.GetDescription())

    // Reset and build second payment
    payment2, err := builder.Reset().
        Amount(200, "EUR").
        Order("ORD_456").
        User("USER_789").
        Build()
    require.NoError(t, err)

    // Verify reset worked
    assert.Equal(t, 200.0, payment2.GetAmount())
    assert.Equal(t, "EUR", payment2.GetCurrency())
    assert.Equal(t, "", payment2.GetDescription()) // Should be empty after reset
}

func TestPaymentRequestBuilder_UnsafeBuild(t *testing.T) {
    // Build without validation
    payment := NewPaymentRequestBuilder().
        Amount(-100, "USD"). // Invalid amount
        Order("").           // Empty order
        User("USER_123").
        BuildUnsafe()

    // Should build without error but not be validated
    assert.Equal(t, -100.0, payment.GetAmount())
    assert.Equal(t, "", payment.GetOrderID())
    assert.False(t, payment.IsValidated())
}

func TestPaymentRequestDirector(t *testing.T) {
    director := NewPaymentRequestDirector(NewPaymentRequestBuilder())

    t.Run("simple payment", func(t *testing.T) {
        payment, err := director.BuildSimplePayment(50.00, "USD", "ORD_001", "USER_123")
        require.NoError(t, err)
        assert.Equal(t, 50.0, payment.GetAmount())
        assert.Equal(t, "USD", payment.GetCurrency())
        assert.Equal(t, "ORD_001", payment.GetOrderID())
        assert.Equal(t, "USER_123", payment.GetUserID())
    })

    t.Run("ecommerce payment", func(t *testing.T) {
        payment, err := director.BuildEcommercePayment(299.99, "USD", "ORD_002", "USER_456", "Product purchase")
        require.NoError(t, err)
        assert.Equal(t, "Product purchase", payment.GetDescription())
        assert.Equal(t, time.Minute*5, payment.GetTimeout())
        assert.Contains(t, payment.GetTags(), "ecommerce")
    })

    t.Run("subscription payment", func(t *testing.T) {
        payment, err := director.BuildSubscriptionPayment(29.99, "USD", "ORD_003", "USER_789", "https://webhook.example.com")
        require.NoError(t, err)
        assert.Equal(t, "Subscription payment", payment.GetDescription())
        assert.NotNil(t, payment.GetWebhook())
        assert.Equal(t, "https://webhook.example.com", payment.GetWebhook().URL)
        assert.NotNil(t, payment.GetRetry())
        assert.Contains(t, payment.GetTags(), "subscription")
        assert.Contains(t, payment.GetTags(), "recurring")
    })

    t.Run("high value payment", func(t *testing.T) {
        payment, err := director.BuildHighValuePayment(10000.00, "USD", "ORD_004", "USER_123", "APPROVER_456")
        require.NoError(t, err)
        assert.Equal(t, "High-value transaction", payment.GetDescription())
        assert.Equal(t, "APPROVER_456", payment.GetMetadata()["approver_id"])
        assert.Equal(t, "high", payment.GetMetadata()["risk_level"])
        assert.Equal(t, time.Hour, payment.GetTimeout())
        assert.Contains(t, payment.GetTags(), "high-value")
        assert.Contains(t, payment.GetTags(), "manual-review")
    })
}

func BenchmarkPaymentRequestBuilder_Build(b *testing.B) {
    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        _, err := NewPaymentRequestBuilder().
            Amount(100.0, "USD").
            Order("ORD_123").
            User("USER_456").
            Description("Benchmark test").
            AddMetadata("key", "value").
            Build()

        if err != nil {
            b.Fatal(err)
        }
    }
}

func BenchmarkPaymentRequestBuilder_BuildComplex(b *testing.B) {
    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        _, err := NewPaymentRequestBuilder().
            Amount(100.0, "USD").
            Order("ORD_123").
            User("USER_456").
            Description("Complex benchmark test").
            AddMetadata("key1", "value1").
            AddMetadata("key2", "value2").
            AddMetadata("key3", "value3").
            WithWebhook("https://api.example.com/webhook").
            WithWebhookHeaders(map[string]string{
                "Authorization": "Bearer token",
                "Content-Type":  "application/json",
            }).
            WithRetry(3, time.Minute).
            WithTimeout(time.Minute * 5).
            AddNote("Note 1").
            AddNote("Note 2").
            AddTags("tag1", "tag2", "tag3").
            Build()

        if err != nil {
            b.Fatal(err)
        }
    }
}
```

## Integration Tips

### 1. Configuration-Driven Builder

```go
type BuilderConfig struct {
    DefaultTimeout time.Duration `yaml:"default_timeout"`
    MaxRetries     int           `yaml:"max_retries"`
    RetryInterval  time.Duration `yaml:"retry_interval"`
    ValidateInputs bool          `yaml:"validate_inputs"`
}

type ConfigurableBuilder struct {
    PaymentRequestBuilder
    config BuilderConfig
}

func (c *ConfigurableBuilder) Build() (*PaymentRequest, error) {
    // Apply defaults from config
    if c.request.timeout == 0 {
        c.WithTimeout(c.config.DefaultTimeout)
    }

    if c.request.retry == nil {
        c.WithRetry(c.config.MaxRetries, c.config.RetryInterval)
    }

    return c.PaymentRequestBuilder.Build()
}
```

### 2. Validation Pipeline Integration

```go
type Validator interface {
    Validate(request *PaymentRequest) error
}

type ValidationPipeline struct {
    validators []Validator
}

type ValidatingBuilder struct {
    PaymentRequestBuilder
    pipeline *ValidationPipeline
}

func (v *ValidatingBuilder) Build() (*PaymentRequest, error) {
    request, err := v.PaymentRequestBuilder.Build()
    if err != nil {
        return nil, err
    }

    for _, validator := range v.pipeline.validators {
        if err := validator.Validate(request); err != nil {
            return nil, fmt.Errorf("validation failed: %w", err)
        }
    }

    return request, nil
}
```

### 3. Event-Driven Builder

```go
type EventPublisher interface {
    Publish(event interface{}) error
}

type EventDrivenBuilder struct {
    PaymentRequestBuilder
    publisher EventPublisher
}

func (e *EventDrivenBuilder) Build() (*PaymentRequest, error) {
    request, err := e.PaymentRequestBuilder.Build()
    if err != nil {
        return nil, err
    }

    // Publish event
    event := PaymentRequestCreated{
        RequestID: request.GetRequestID(),
        Amount:    request.GetAmount(),
        Currency:  request.GetCurrency(),
        CreatedAt: request.GetCreatedAt(),
    }

    if err := e.publisher.Publish(event); err != nil {
        // Log error but don't fail the build
        log.Printf("Failed to publish event: %v", err)
    }

    return request, nil
}
```

### 4. Caching Integration

```go
type CachingBuilder struct {
    PaymentRequestBuilder
    cache Cache
}

func (c *CachingBuilder) Build() (*PaymentRequest, error) {
    // Generate cache key from request data
    key := c.generateCacheKey()

    // Check cache first
    if cached, found := c.cache.Get(key); found {
        return cached.(*PaymentRequest), nil
    }

    // Build and cache
    request, err := c.PaymentRequestBuilder.Build()
    if err != nil {
        return nil, err
    }

    c.cache.Set(key, request, time.Minute*5)
    return request, nil
}
```

## Common Interview Questions

### 1. **How does Builder pattern solve the telescoping constructor problem?**

**Answer:**
The telescoping constructor problem occurs when you have multiple constructors with different parameter combinations:

```go
// Telescoping constructor anti-pattern
func NewPaymentRequest(amount float64, currency string) *PaymentRequest
func NewPaymentRequestWithOrder(amount float64, currency, orderID string) *PaymentRequest
func NewPaymentRequestWithUser(amount float64, currency, orderID, userID string) *PaymentRequest
func NewPaymentRequestWithDescription(amount float64, currency, orderID, userID, description string) *PaymentRequest
// ... many more variations
```

**Problems:**

- Exponential growth of constructors
- Parameter order confusion
- Hard to remember which constructor to use
- No way to set only some optional parameters

**Builder solution:**

```go
// Clean, readable, flexible
payment := NewPaymentRequestBuilder().
    Amount(100, "USD").
    Order("ORD_123").
    User("USER_456").
    Description("Optional description").
    Build()
```

### 2. **When would you use Builder vs Factory patterns?**

**Answer:**

| Scenario                             | Pattern          | Reason                                         |
| ------------------------------------ | ---------------- | ---------------------------------------------- |
| Object with many optional parameters | Builder          | Step-by-step construction with optional fields |
| Object creation varies by type       | Factory Method   | Different object types, same interface         |
| Family of related objects            | Abstract Factory | Consistent object families                     |
| Complex object with validation       | Builder          | Validation during construction                 |
| Simple object creation               | Factory Method   | Less overhead                                  |

**Example:**

```go
// Builder - complex configuration
payment := NewPaymentRequestBuilder().
    Amount(100, "USD").
    // ... many optional fields
    Build()

// Factory - different types
processor := ProcessorFactory.Create("stripe") // or "razorpay"
```

### 3. **How do you implement immutability with Builder pattern?**

**Answer:**

1. **Private fields** in the product
2. **Deep copying** in the builder
3. **Validation** before returning

```go
type PaymentRequest struct {
    amount   float64  // private fields
    currency string
    metadata map[string]string
}

// Getters only, no setters
func (p *PaymentRequest) GetAmount() float64 { return p.amount }

func (b *Builder) Build() (*PaymentRequest, error) {
    // Deep copy to ensure immutability
    metadata := make(map[string]string)
    for k, v := range b.request.metadata {
        metadata[k] = v
    }

    return &PaymentRequest{
        amount:   b.request.amount,
        currency: b.request.currency,
        metadata: metadata, // new map, not shared
    }, nil
}
```

### 4. **How do you handle validation in Builder pattern?**

**Answer:**
Multiple validation strategies:

1. **Immediate validation** (fail fast):

```go
func (b *Builder) Amount(amount float64, currency string) Builder {
    if amount <= 0 {
        b.errors = append(b.errors, errors.New("amount must be positive"))
    }
    b.request.amount = amount
    return b
}
```

2. **Build-time validation**:

```go
func (b *Builder) Build() (*PaymentRequest, error) {
    if len(b.errors) > 0 {
        return nil, fmt.Errorf("validation errors: %v", b.errors)
    }

    if err := b.request.Validate(); err != nil {
        return nil, err
    }

    return b.request, nil
}
```

3. **Pipeline validation**:

```go
type ValidationRule func(*PaymentRequest) error

func (b *Builder) AddValidationRule(rule ValidationRule) Builder {
    b.validationRules = append(b.validationRules, rule)
    return b
}
```

### 5. **How do you implement Builder pattern with generics in Go?**

**Answer:**

```go
// Generic builder interface
type Builder[T any] interface {
    Build() (T, error)
    Reset() Builder[T]
}

// Generic fluent builder
type FluentBuilder[T any] struct {
    product T
    errors  []error
}

func NewBuilder[T any]() *FluentBuilder[T] {
    return &FluentBuilder[T]{
        errors: make([]error, 0),
    }
}

func (b *FluentBuilder[T]) Build() (T, error) {
    if len(b.errors) > 0 {
        var zero T
        return zero, fmt.Errorf("validation errors: %v", b.errors)
    }
    return b.product, nil
}

// Usage
type PaymentRequestBuilder struct {
    *FluentBuilder[*PaymentRequest]
}

func NewPaymentRequestBuilder() *PaymentRequestBuilder {
    return &PaymentRequestBuilder{
        FluentBuilder: NewBuilder[*PaymentRequest](),
    }
}
```
