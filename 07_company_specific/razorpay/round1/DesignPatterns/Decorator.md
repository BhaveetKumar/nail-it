---
# Auto-generated front matter
Title: Decorator
LastUpdated: 2025-11-06T20:45:58.524275
Tags: []
Status: draft
---

# Decorator Pattern

## Pattern Name & Intent

**Decorator** is a structural design pattern that lets you attach new behaviors to objects by placing these objects inside special wrapper objects that contain the behaviors. It provides a flexible alternative to subclassing for extending functionality.

**Key Intent:**

- Add new functionality to objects dynamically without altering their structure
- Provide a flexible alternative to inheritance for extending behavior
- Allow behavior to be extended at runtime
- Follow the Open-Closed Principle (open for extension, closed for modification)
- Compose behaviors by stacking decorators

## When to Use

**Use Decorator when:**

1. **Runtime Extension**: Need to add behavior to objects at runtime
2. **Flexible Composition**: Want to combine multiple behaviors flexibly
3. **Alternative to Inheritance**: Inheritance would create too many subclasses
4. **Single Responsibility**: Each decorator handles one concern
5. **Optional Features**: Features that can be optionally applied
6. **Cross-cutting Concerns**: Logging, caching, authentication, etc.
7. **Middleware Patterns**: Request/response processing pipelines

**Don't use when:**

- Simple inheritance is sufficient
- The decorator interface becomes too complex
- You need to remove decorators frequently
- The order of decoration doesn't matter
- Performance is critical (adds method call overhead)

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Processing Pipeline

```go
// Base payment processor
type PaymentProcessor interface {
    ProcessPayment(ctx context.Context, payment *Payment) (*PaymentResult, error)
}

type BasicPaymentProcessor struct {
    gateway PaymentGateway
}

func (b *BasicPaymentProcessor) ProcessPayment(ctx context.Context, payment *Payment) (*PaymentResult, error) {
    return b.gateway.Process(ctx, payment)
}

// Logging decorator
type LoggingPaymentProcessor struct {
    processor PaymentProcessor
    logger    *zap.Logger
}

func (l *LoggingPaymentProcessor) ProcessPayment(ctx context.Context, payment *Payment) (*PaymentResult, error) {
    l.logger.Info("Processing payment",
        zap.String("payment_id", payment.ID),
        zap.String("amount", payment.Amount.String()))

    result, err := l.processor.ProcessPayment(ctx, payment)

    if err != nil {
        l.logger.Error("Payment processing failed",
            zap.String("payment_id", payment.ID),
            zap.Error(err))
    } else {
        l.logger.Info("Payment processed successfully",
            zap.String("payment_id", payment.ID),
            zap.String("transaction_id", result.TransactionID))
    }

    return result, err
}

// Validation decorator
type ValidationPaymentProcessor struct {
    processor PaymentProcessor
    validator PaymentValidator
}

func (v *ValidationPaymentProcessor) ProcessPayment(ctx context.Context, payment *Payment) (*PaymentResult, error) {
    if err := v.validator.Validate(payment); err != nil {
        return nil, fmt.Errorf("payment validation failed: %w", err)
    }

    return v.processor.ProcessPayment(ctx, payment)
}

// Fraud detection decorator
type FraudDetectionPaymentProcessor struct {
    processor   PaymentProcessor
    fraudEngine FraudDetectionEngine
}

func (f *FraudDetectionPaymentProcessor) ProcessPayment(ctx context.Context, payment *Payment) (*PaymentResult, error) {
    riskScore, err := f.fraudEngine.CalculateRisk(ctx, payment)
    if err != nil {
        return nil, fmt.Errorf("fraud detection failed: %w", err)
    }

    if riskScore > 0.8 {
        return nil, fmt.Errorf("payment blocked due to high fraud risk: %.2f", riskScore)
    }

    result, err := f.processor.ProcessPayment(ctx, payment)
    if err == nil {
        result.RiskScore = riskScore
    }

    return result, err
}

// Retry decorator
type RetryPaymentProcessor struct {
    processor   PaymentProcessor
    maxRetries  int
    retryDelay  time.Duration
}

func (r *RetryPaymentProcessor) ProcessPayment(ctx context.Context, payment *Payment) (*PaymentResult, error) {
    var lastErr error

    for attempt := 0; attempt <= r.maxRetries; attempt++ {
        if attempt > 0 {
            select {
            case <-ctx.Done():
                return nil, ctx.Err()
            case <-time.After(r.retryDelay):
            }
        }

        result, err := r.processor.ProcessPayment(ctx, payment)
        if err == nil {
            return result, nil
        }

        lastErr = err

        // Don't retry for certain error types
        if !isRetryableError(err) {
            break
        }
    }

    return nil, fmt.Errorf("payment failed after %d attempts: %w", r.maxRetries+1, lastErr)
}

// Usage - compose decorators
func CreatePaymentProcessor(gateway PaymentGateway, logger *zap.Logger) PaymentProcessor {
    processor := &BasicPaymentProcessor{gateway: gateway}

    // Wrap with decorators
    processor = &ValidationPaymentProcessor{
        processor: processor,
        validator: NewPaymentValidator(),
    }

    processor = &FraudDetectionPaymentProcessor{
        processor:   processor,
        fraudEngine: NewFraudDetectionEngine(),
    }

    processor = &RetryPaymentProcessor{
        processor:   processor,
        maxRetries:  3,
        retryDelay:  time.Second,
    }

    processor = &LoggingPaymentProcessor{
        processor: processor,
        logger:    logger,
    }

    return processor
}
```

### 2. Account Balance Decorators

```go
// Base account interface
type Account interface {
    GetBalance(ctx context.Context) (decimal.Decimal, error)
    Debit(ctx context.Context, amount decimal.Decimal, description string) error
    Credit(ctx context.Context, amount decimal.Decimal, description string) error
    GetAccountID() string
}

type BasicAccount struct {
    accountID string
    balance   decimal.Decimal
    mutex     sync.RWMutex
}

func (b *BasicAccount) GetBalance(ctx context.Context) (decimal.Decimal, error) {
    b.mutex.RLock()
    defer b.mutex.RUnlock()
    return b.balance, nil
}

func (b *BasicAccount) Debit(ctx context.Context, amount decimal.Decimal, description string) error {
    b.mutex.Lock()
    defer b.mutex.Unlock()

    if b.balance.LessThan(amount) {
        return fmt.Errorf("insufficient balance")
    }

    b.balance = b.balance.Sub(amount)
    return nil
}

// Overdraft protection decorator
type OverdraftProtectionAccount struct {
    Account
    overdraftLimit decimal.Decimal
}

func (o *OverdraftProtectionAccount) Debit(ctx context.Context, amount decimal.Decimal, description string) error {
    balance, err := o.Account.GetBalance(ctx)
    if err != nil {
        return err
    }

    availableBalance := balance.Add(o.overdraftLimit)
    if availableBalance.LessThan(amount) {
        return fmt.Errorf("debit amount exceeds available balance including overdraft limit")
    }

    return o.Account.Debit(ctx, amount, description)
}

// Interest calculation decorator
type InterestBearingAccount struct {
    Account
    interestRate decimal.Decimal
    lastUpdate   time.Time
}

func (i *InterestBearingAccount) GetBalance(ctx context.Context) (decimal.Decimal, error) {
    // Calculate and add accrued interest
    if err := i.accrueInterest(); err != nil {
        return decimal.Zero, err
    }

    return i.Account.GetBalance(ctx)
}

func (i *InterestBearingAccount) accrueInterest() error {
    now := time.Now()
    daysSinceUpdate := now.Sub(i.lastUpdate).Hours() / 24

    if daysSinceUpdate > 0 {
        balance, err := i.Account.GetBalance(context.Background())
        if err != nil {
            return err
        }

        dailyRate := i.interestRate.Div(decimal.NewFromInt(365))
        interest := balance.Mul(dailyRate).Mul(decimal.NewFromFloat(daysSinceUpdate))

        err = i.Account.Credit(context.Background(), interest, "Accrued interest")
        if err != nil {
            return err
        }

        i.lastUpdate = now
    }

    return nil
}

// Notification decorator
type NotificationAccount struct {
    Account
    notifier NotificationService
}

func (n *NotificationAccount) Debit(ctx context.Context, amount decimal.Decimal, description string) error {
    err := n.Account.Debit(ctx, amount, description)
    if err == nil {
        balance, _ := n.Account.GetBalance(ctx)
        n.notifier.SendNotification(NotificationRequest{
            AccountID: n.GetAccountID(),
            Type:      "DEBIT",
            Amount:    amount,
            Balance:   balance,
            Message:   fmt.Sprintf("Debit: %s - %s", amount, description),
        })
    }
    return err
}
```

### 3. API Endpoint Decorators

```go
// HTTP handler interface
type Handler interface {
    ServeHTTP(ctx context.Context, request *HTTPRequest) (*HTTPResponse, error)
}

type BaseHandler struct {
    handlerFunc func(ctx context.Context, request *HTTPRequest) (*HTTPResponse, error)
}

func (b *BaseHandler) ServeHTTP(ctx context.Context, request *HTTPRequest) (*HTTPResponse, error) {
    return b.handlerFunc(ctx, request)
}

// Authentication decorator
type AuthenticationHandler struct {
    Handler
    authService AuthenticationService
}

func (a *AuthenticationHandler) ServeHTTP(ctx context.Context, request *HTTPRequest) (*HTTPResponse, error) {
    token := request.Headers["Authorization"]
    if token == "" {
        return &HTTPResponse{
            StatusCode: 401,
            Body:       "Missing authorization token",
        }, nil
    }

    user, err := a.authService.ValidateToken(ctx, token)
    if err != nil {
        return &HTTPResponse{
            StatusCode: 401,
            Body:       "Invalid token",
        }, nil
    }

    // Add user to context
    ctx = context.WithValue(ctx, "user", user)

    return a.Handler.ServeHTTP(ctx, request)
}

// Rate limiting decorator
type RateLimitingHandler struct {
    Handler
    limiter RateLimiter
}

func (r *RateLimitingHandler) ServeHTTP(ctx context.Context, request *HTTPRequest) (*HTTPResponse, error) {
    clientID := getClientID(request)

    allowed, err := r.limiter.Allow(ctx, clientID)
    if err != nil {
        return &HTTPResponse{
            StatusCode: 500,
            Body:       "Rate limiting error",
        }, err
    }

    if !allowed {
        return &HTTPResponse{
            StatusCode: 429,
            Body:       "Rate limit exceeded",
        }, nil
    }

    return r.Handler.ServeHTTP(ctx, request)
}

// Caching decorator
type CachingHandler struct {
    Handler
    cache Cache
    ttl   time.Duration
}

func (c *CachingHandler) ServeHTTP(ctx context.Context, request *HTTPRequest) (*HTTPResponse, error) {
    if request.Method != "GET" {
        return c.Handler.ServeHTTP(ctx, request)
    }

    cacheKey := generateCacheKey(request)

    // Try to get from cache
    if cached, err := c.cache.Get(cacheKey); err == nil {
        return cached.(*HTTPResponse), nil
    }

    // Not in cache, process request
    response, err := c.Handler.ServeHTTP(ctx, request)
    if err != nil {
        return response, err
    }

    // Cache successful responses
    if response.StatusCode == 200 {
        c.cache.Set(cacheKey, response, c.ttl)
    }

    return response, nil
}

// Metrics decorator
type MetricsHandler struct {
    Handler
    metrics MetricsCollector
}

func (m *MetricsHandler) ServeHTTP(ctx context.Context, request *HTTPRequest) (*HTTPResponse, error) {
    start := time.Now()

    response, err := m.Handler.ServeHTTP(ctx, request)

    duration := time.Since(start)

    m.metrics.RecordRequestDuration(request.Method, request.Path, duration)
    if response != nil {
        m.metrics.RecordResponseStatus(response.StatusCode)
    }
    if err != nil {
        m.metrics.RecordError(request.Method, request.Path)
    }

    return response, err
}
```

## Go Implementation

```go
package main

import (
    "context"
    "fmt"
    "time"
    "sync"
    "github.com/shopspring/decimal"
    "go.uber.org/zap"
)

// Base component interface
type NotificationService interface {
    SendNotification(ctx context.Context, message NotificationMessage) error
}

type NotificationMessage struct {
    ID          string
    RecipientID string
    Subject     string
    Body        string
    Type        string
    Priority    int
    Metadata    map[string]interface{}
}

// Basic notification service (concrete component)
type BasicNotificationService struct {
    emailSender EmailSender
    smsSender   SMSSender
}

func NewBasicNotificationService(emailSender EmailSender, smsSender SMSSender) *BasicNotificationService {
    return &BasicNotificationService{
        emailSender: emailSender,
        smsSender:   smsSender,
    }
}

func (b *BasicNotificationService) SendNotification(ctx context.Context, message NotificationMessage) error {
    switch message.Type {
    case "EMAIL":
        return b.emailSender.SendEmail(ctx, EmailMessage{
            To:      message.RecipientID,
            Subject: message.Subject,
            Body:    message.Body,
        })
    case "SMS":
        return b.smsSender.SendSMS(ctx, SMSMessage{
            To:   message.RecipientID,
            Body: message.Body,
        })
    default:
        return fmt.Errorf("unsupported notification type: %s", message.Type)
    }
}

// Decorator base
type NotificationDecorator struct {
    NotificationService
}

func (n *NotificationDecorator) SendNotification(ctx context.Context, message NotificationMessage) error {
    return n.NotificationService.SendNotification(ctx, message)
}

// Logging decorator
type LoggingNotificationDecorator struct {
    *NotificationDecorator
    logger *zap.Logger
}

func NewLoggingNotificationDecorator(service NotificationService, logger *zap.Logger) *LoggingNotificationDecorator {
    return &LoggingNotificationDecorator{
        NotificationDecorator: &NotificationDecorator{NotificationService: service},
        logger:               logger,
    }
}

func (l *LoggingNotificationDecorator) SendNotification(ctx context.Context, message NotificationMessage) error {
    l.logger.Info("Sending notification",
        zap.String("id", message.ID),
        zap.String("type", message.Type),
        zap.String("recipient", message.RecipientID),
        zap.Int("priority", message.Priority))

    start := time.Now()
    err := l.NotificationDecorator.SendNotification(ctx, message)
    duration := time.Since(start)

    if err != nil {
        l.logger.Error("Failed to send notification",
            zap.String("id", message.ID),
            zap.Error(err),
            zap.Duration("duration", duration))
    } else {
        l.logger.Info("Successfully sent notification",
            zap.String("id", message.ID),
            zap.Duration("duration", duration))
    }

    return err
}

// Retry decorator
type RetryNotificationDecorator struct {
    *NotificationDecorator
    maxRetries int
    backoff    BackoffStrategy
}

type BackoffStrategy interface {
    NextDelay(attempt int) time.Duration
}

type ExponentialBackoff struct {
    baseDelay time.Duration
    maxDelay  time.Duration
}

func (e *ExponentialBackoff) NextDelay(attempt int) time.Duration {
    delay := e.baseDelay * time.Duration(1<<uint(attempt))
    if delay > e.maxDelay {
        delay = e.maxDelay
    }
    return delay
}

func NewRetryNotificationDecorator(service NotificationService, maxRetries int, backoff BackoffStrategy) *RetryNotificationDecorator {
    return &RetryNotificationDecorator{
        NotificationDecorator: &NotificationDecorator{NotificationService: service},
        maxRetries:           maxRetries,
        backoff:              backoff,
    }
}

func (r *RetryNotificationDecorator) SendNotification(ctx context.Context, message NotificationMessage) error {
    var lastErr error

    for attempt := 0; attempt <= r.maxRetries; attempt++ {
        if attempt > 0 {
            delay := r.backoff.NextDelay(attempt - 1)
            select {
            case <-ctx.Done():
                return ctx.Err()
            case <-time.After(delay):
            }
        }

        err := r.NotificationDecorator.SendNotification(ctx, message)
        if err == nil {
            return nil
        }

        lastErr = err

        // Don't retry for certain error types
        if !isRetryableError(err) {
            break
        }
    }

    return fmt.Errorf("notification failed after %d attempts: %w", r.maxRetries+1, lastErr)
}

// Rate limiting decorator
type RateLimitingNotificationDecorator struct {
    *NotificationDecorator
    limiter *TokenBucketLimiter
    mu      sync.Mutex
}

type TokenBucketLimiter struct {
    capacity    int
    tokens      int
    refillRate  int // tokens per second
    lastRefill  time.Time
    mu          sync.Mutex
}

func NewTokenBucketLimiter(capacity, refillRate int) *TokenBucketLimiter {
    return &TokenBucketLimiter{
        capacity:   capacity,
        tokens:     capacity,
        refillRate: refillRate,
        lastRefill: time.Now(),
    }
}

func (t *TokenBucketLimiter) Allow() bool {
    t.mu.Lock()
    defer t.mu.Unlock()

    now := time.Now()
    elapsed := now.Sub(t.lastRefill)

    // Refill tokens
    tokensToAdd := int(elapsed.Seconds()) * t.refillRate
    t.tokens = min(t.capacity, t.tokens+tokensToAdd)
    t.lastRefill = now

    if t.tokens > 0 {
        t.tokens--
        return true
    }

    return false
}

func NewRateLimitingNotificationDecorator(service NotificationService, limiter *TokenBucketLimiter) *RateLimitingNotificationDecorator {
    return &RateLimitingNotificationDecorator{
        NotificationDecorator: &NotificationDecorator{NotificationService: service},
        limiter:              limiter,
    }
}

func (r *RateLimitingNotificationDecorator) SendNotification(ctx context.Context, message NotificationMessage) error {
    if !r.limiter.Allow() {
        return fmt.Errorf("rate limit exceeded for notification service")
    }

    return r.NotificationDecorator.SendNotification(ctx, message)
}

// Circuit breaker decorator
type CircuitBreakerNotificationDecorator struct {
    *NotificationDecorator
    circuitBreaker *CircuitBreaker
}

type CircuitBreakerState int

const (
    CircuitClosed CircuitBreakerState = iota
    CircuitOpen
    CircuitHalfOpen
)

type CircuitBreaker struct {
    failureThreshold int
    timeout          time.Duration
    state            CircuitBreakerState
    failureCount     int
    lastFailureTime  time.Time
    successCount     int
    mu               sync.RWMutex
}

func NewCircuitBreaker(failureThreshold int, timeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        failureThreshold: failureThreshold,
        timeout:          timeout,
        state:            CircuitClosed,
    }
}

func (c *CircuitBreaker) Call(fn func() error) error {
    c.mu.Lock()
    defer c.mu.Unlock()

    switch c.state {
    case CircuitOpen:
        if time.Since(c.lastFailureTime) > c.timeout {
            c.state = CircuitHalfOpen
            c.successCount = 0
        } else {
            return fmt.Errorf("circuit breaker is open")
        }
    case CircuitHalfOpen:
        // Allow limited calls to test if service is back
    case CircuitClosed:
        // Normal operation
    }

    err := fn()

    if err != nil {
        c.failureCount++
        c.lastFailureTime = time.Now()

        if c.state == CircuitHalfOpen {
            c.state = CircuitOpen
        } else if c.failureCount >= c.failureThreshold {
            c.state = CircuitOpen
        }

        return err
    }

    // Success
    if c.state == CircuitHalfOpen {
        c.successCount++
        if c.successCount >= 3 { // Require several successes to close
            c.state = CircuitClosed
            c.failureCount = 0
        }
    } else {
        c.failureCount = 0
    }

    return nil
}

func NewCircuitBreakerNotificationDecorator(service NotificationService, circuitBreaker *CircuitBreaker) *CircuitBreakerNotificationDecorator {
    return &CircuitBreakerNotificationDecorator{
        NotificationDecorator: &NotificationDecorator{NotificationService: service},
        circuitBreaker:       circuitBreaker,
    }
}

func (c *CircuitBreakerNotificationDecorator) SendNotification(ctx context.Context, message NotificationMessage) error {
    return c.circuitBreaker.Call(func() error {
        return c.NotificationDecorator.SendNotification(ctx, message)
    })
}

// Encryption decorator
type EncryptionNotificationDecorator struct {
    *NotificationDecorator
    encryptor Encryptor
}

type Encryptor interface {
    Encrypt(data string) (string, error)
    Decrypt(data string) (string, error)
}

type AESEncryptor struct {
    key []byte
}

func NewAESEncryptor(key []byte) *AESEncryptor {
    return &AESEncryptor{key: key}
}

func (a *AESEncryptor) Encrypt(data string) (string, error) {
    // Simplified encryption implementation
    return fmt.Sprintf("encrypted_%s", data), nil
}

func (a *AESEncryptor) Decrypt(data string) (string, error) {
    // Simplified decryption implementation
    if strings.HasPrefix(data, "encrypted_") {
        return strings.TrimPrefix(data, "encrypted_"), nil
    }
    return data, nil
}

func NewEncryptionNotificationDecorator(service NotificationService, encryptor Encryptor) *EncryptionNotificationDecorator {
    return &EncryptionNotificationDecorator{
        NotificationDecorator: &NotificationDecorator{NotificationService: service},
        encryptor:            encryptor,
    }
}

func (e *EncryptionNotificationDecorator) SendNotification(ctx context.Context, message NotificationMessage) error {
    // Encrypt sensitive content
    encryptedSubject, err := e.encryptor.Encrypt(message.Subject)
    if err != nil {
        return fmt.Errorf("failed to encrypt subject: %w", err)
    }

    encryptedBody, err := e.encryptor.Encrypt(message.Body)
    if err != nil {
        return fmt.Errorf("failed to encrypt body: %w", err)
    }

    encryptedMessage := message
    encryptedMessage.Subject = encryptedSubject
    encryptedMessage.Body = encryptedBody

    return e.NotificationDecorator.SendNotification(ctx, encryptedMessage)
}

// Priority queue decorator
type PriorityQueueNotificationDecorator struct {
    *NotificationDecorator
    highPriorityQueue chan NotificationMessage
    normalQueue       chan NotificationMessage
    lowPriorityQueue  chan NotificationMessage
    quit              chan struct{}
    wg                sync.WaitGroup
}

func NewPriorityQueueNotificationDecorator(service NotificationService) *PriorityQueueNotificationDecorator {
    decorator := &PriorityQueueNotificationDecorator{
        NotificationDecorator: &NotificationDecorator{NotificationService: service},
        highPriorityQueue:    make(chan NotificationMessage, 100),
        normalQueue:          make(chan NotificationMessage, 1000),
        lowPriorityQueue:     make(chan NotificationMessage, 10000),
        quit:                 make(chan struct{}),
    }

    decorator.startWorkers()
    return decorator
}

func (p *PriorityQueueNotificationDecorator) startWorkers() {
    // Start worker goroutines
    for i := 0; i < 3; i++ {
        p.wg.Add(1)
        go p.worker()
    }
}

func (p *PriorityQueueNotificationDecorator) worker() {
    defer p.wg.Done()

    for {
        select {
        case <-p.quit:
            return
        case message := <-p.highPriorityQueue:
            p.processMessage(message)
        case message := <-p.normalQueue:
            p.processMessage(message)
        case message := <-p.lowPriorityQueue:
            p.processMessage(message)
        }
    }
}

func (p *PriorityQueueNotificationDecorator) processMessage(message NotificationMessage) {
    ctx := context.Background()
    err := p.NotificationDecorator.SendNotification(ctx, message)
    if err != nil {
        // Handle error (could retry, log, etc.)
        fmt.Printf("Failed to send notification %s: %v\n", message.ID, err)
    }
}

func (p *PriorityQueueNotificationDecorator) SendNotification(ctx context.Context, message NotificationMessage) error {
    // Queue message based on priority
    switch {
    case message.Priority >= 8: // High priority
        select {
        case p.highPriorityQueue <- message:
            return nil
        default:
            return fmt.Errorf("high priority queue is full")
        }
    case message.Priority >= 4: // Normal priority
        select {
        case p.normalQueue <- message:
            return nil
        default:
            return fmt.Errorf("normal priority queue is full")
        }
    default: // Low priority
        select {
        case p.lowPriorityQueue <- message:
            return nil
        default:
            return fmt.Errorf("low priority queue is full")
        }
    }
}

func (p *PriorityQueueNotificationDecorator) Shutdown() {
    close(p.quit)
    p.wg.Wait()
}

// Helper interfaces and implementations
type EmailSender interface {
    SendEmail(ctx context.Context, message EmailMessage) error
}

type SMSSender interface {
    SendSMS(ctx context.Context, message SMSMessage) error
}

type EmailMessage struct {
    To      string
    Subject string
    Body    string
}

type SMSMessage struct {
    To   string
    Body string
}

type MockEmailSender struct{}

func (m *MockEmailSender) SendEmail(ctx context.Context, message EmailMessage) error {
    fmt.Printf("Sending email to %s: %s\n", message.To, message.Subject)
    return nil
}

type MockSMSSender struct{}

func (m *MockSMSSender) SendSMS(ctx context.Context, message SMSMessage) error {
    fmt.Printf("Sending SMS to %s: %s\n", message.To, message.Body)
    return nil
}

// Helper functions
func isRetryableError(err error) bool {
    // Simplified: retry for network errors, not for validation errors
    return !strings.Contains(err.Error(), "validation") &&
           !strings.Contains(err.Error(), "invalid")
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

// Example usage
func main() {
    fmt.Println("=== Decorator Pattern Demo ===\n")

    // Create logger
    logger, _ := zap.NewDevelopment()
    defer logger.Sync()

    // Create basic notification service
    emailSender := &MockEmailSender{}
    smsSender := &MockSMSSender{}
    basicService := NewBasicNotificationService(emailSender, smsSender)

    // Create decorators
    var service NotificationService = basicService

    // Add logging
    service = NewLoggingNotificationDecorator(service, logger)

    // Add retry functionality
    backoff := &ExponentialBackoff{
        baseDelay: 100 * time.Millisecond,
        maxDelay:  5 * time.Second,
    }
    service = NewRetryNotificationDecorator(service, 3, backoff)

    // Add rate limiting
    rateLimiter := NewTokenBucketLimiter(10, 2) // 10 tokens, refill 2 per second
    service = NewRateLimitingNotificationDecorator(service, rateLimiter)

    // Add circuit breaker
    circuitBreaker := NewCircuitBreaker(3, 30*time.Second)
    service = NewCircuitBreakerNotificationDecorator(service, circuitBreaker)

    // Add encryption
    encryptor := NewAESEncryptor([]byte("secret-key-32-characters-long"))
    service = NewEncryptionNotificationDecorator(service, encryptor)

    // Create test messages
    messages := []NotificationMessage{
        {
            ID:          "msg-1",
            RecipientID: "user@example.com",
            Subject:     "Welcome!",
            Body:        "Welcome to our platform!",
            Type:        "EMAIL",
            Priority:    5,
        },
        {
            ID:          "msg-2",
            RecipientID: "+1234567890",
            Subject:     "",
            Body:        "Your verification code is 123456",
            Type:        "SMS",
            Priority:    8,
        },
        {
            ID:          "msg-3",
            RecipientID: "admin@example.com",
            Subject:     "System Alert",
            Body:        "High CPU usage detected",
            Type:        "EMAIL",
            Priority:    9,
        },
    }

    // Send notifications
    ctx := context.Background()

    fmt.Println("Sending notifications with decorated service:")
    for _, message := range messages {
        err := service.SendNotification(ctx, message)
        if err != nil {
            fmt.Printf("Error sending notification %s: %v\n", message.ID, err)
        } else {
            fmt.Printf("Successfully queued notification %s\n", message.ID)
        }
        time.Sleep(100 * time.Millisecond) // Small delay between messages
    }

    // Test rate limiting
    fmt.Println("\nTesting rate limiting:")
    for i := 0; i < 15; i++ {
        message := NotificationMessage{
            ID:          fmt.Sprintf("rate-test-%d", i),
            RecipientID: "test@example.com",
            Subject:     "Rate Test",
            Body:        "Testing rate limiting",
            Type:        "EMAIL",
            Priority:    1,
        }

        err := service.SendNotification(ctx, message)
        if err != nil {
            fmt.Printf("Message %d: Rate limited - %v\n", i+1, err)
        } else {
            fmt.Printf("Message %d: Sent successfully\n", i+1)
        }
    }

    fmt.Println("\n=== Decorator Pattern Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Interface-based Decorators**

```go
type Component interface {
    Operation() string
}

type ConcreteComponent struct{}

func (c *ConcreteComponent) Operation() string {
    return "ConcreteComponent"
}

type Decorator interface {
    Component
    SetComponent(Component)
}

type ConcreteDecorator struct {
    component Component
}

func (d *ConcreteDecorator) SetComponent(c Component) {
    d.component = c
}

func (d *ConcreteDecorator) Operation() string {
    return "ConcreteDecorator(" + d.component.Operation() + ")"
}
```

2. **Functional Decorators**

```go
type HandlerFunc func(context.Context, *Request) (*Response, error)

type Middleware func(HandlerFunc) HandlerFunc

func LoggingMiddleware(logger *zap.Logger) Middleware {
    return func(next HandlerFunc) HandlerFunc {
        return func(ctx context.Context, req *Request) (*Response, error) {
            logger.Info("Processing request", zap.String("path", req.Path))
            resp, err := next(ctx, req)
            if err != nil {
                logger.Error("Request failed", zap.Error(err))
            }
            return resp, err
        }
    }
}

func AuthMiddleware(authService AuthService) Middleware {
    return func(next HandlerFunc) HandlerFunc {
        return func(ctx context.Context, req *Request) (*Response, error) {
            if !authService.IsAuthenticated(req) {
                return &Response{Status: 401}, nil
            }
            return next(ctx, req)
        }
    }
}

// Chain middlewares
func ChainMiddlewares(handler HandlerFunc, middlewares ...Middleware) HandlerFunc {
    for i := len(middlewares) - 1; i >= 0; i-- {
        handler = middlewares[i](../../../../08_interview_prep/practice/handler)
    }
    return handler
}
```

3. **Type-safe Decorators with Generics**

```go
type Processor[T any] interface {
    Process(ctx context.Context, input T) (T, error)
}

type ProcessorDecorator[T any] struct {
    wrapped Processor[T]
}

func (p *ProcessorDecorator[T]) Process(ctx context.Context, input T) (T, error) {
    return p.wrapped.Process(ctx, input)
}

type LoggingProcessor[T any] struct {
    ProcessorDecorator[T]
    logger *zap.Logger
}

func NewLoggingProcessor[T any](processor Processor[T], logger *zap.Logger/) *LoggingProcessor[T] {
    return &LoggingProcessor[T]{
        ProcessorDecorator: ProcessorDecorator[T]{wrapped: processor},
        logger:            logger,
    }
}

func (l *LoggingProcessor[T]) Process(ctx context.Context, input T) (T, error) {
    l.logger.Info("Processing input")
    result, err := l.ProcessorDecorator.Process(ctx, input)
    if err != nil {
        l.logger.Error("Processing failed", zap.Error(err))
    }
    return result, err
}
```

### Trade-offs

**Pros:**

- **Runtime Composition**: Add behaviors dynamically
- **Single Responsibility**: Each decorator has one concern
- **Open-Closed Principle**: Extend without modifying existing code
- **Flexible Combinations**: Mix and match decorators
- **Inheritance Alternative**: Avoid explosion of subclasses

**Cons:**

- **Complexity**: Can create deep object hierarchies
- **Performance**: Additional method calls and object creation
- **Debugging**: Stack traces can become complex
- **Interface Pollution**: Component interface may become broad
- **Order Dependency**: Decorator order can affect behavior

**When to Choose Decorator vs Alternatives:**

| Scenario               | Pattern                 | Reason                       |
| ---------------------- | ----------------------- | ---------------------------- |
| Cross-cutting concerns | Decorator               | Clean separation of concerns |
| Pipeline processing    | Chain of Responsibility | Sequential processing        |
| Object state changes   | State                   | Behavior changes with state  |
| Algorithm selection    | Strategy                | Different algorithms         |
| Single enhancement     | Inheritance             | Simpler for one enhancement  |

## Integration Tips

### 1. Factory Integration

```go
type DecoratorFactory interface {
    CreateDecorator(service NotificationService, config DecoratorConfig) NotificationService
}

type DecoratorConfig struct {
    Type   string
    Params map[string]interface{}
}

type NotificationDecoratorFactory struct {
    logger         *zap.Logger
    rateLimiter    *TokenBucketLimiter
    circuitBreaker *CircuitBreaker
}

func (f *NotificationDecoratorFactory) CreateDecorator(service NotificationService, config DecoratorConfig) NotificationService {
    switch config.Type {
    case "logging":
        return NewLoggingNotificationDecorator(service, f.logger)
    case "retry":
        maxRetries := config.Params["maxRetries"].(int)
        return NewRetryNotificationDecorator(service, maxRetries, &ExponentialBackoff{})
    case "rateLimit":
        return NewRateLimitingNotificationDecorator(service, f.rateLimiter)
    default:
        return service
    }
}
```

### 2. Builder Pattern Integration

```go
type NotificationServiceBuilder struct {
    baseService NotificationService
    decorators  []DecoratorConfig
}

func NewNotificationServiceBuilder(baseService NotificationService) *NotificationServiceBuilder {
    return &NotificationServiceBuilder{
        baseService: baseService,
        decorators:  make([]DecoratorConfig, 0),
    }
}

func (b *NotificationServiceBuilder) WithLogging(logger *zap.Logger) *NotificationServiceBuilder {
    b.decorators = append(b.decorators, DecoratorConfig{
        Type:   "logging",
        Params: map[string]interface{}{"logger": logger},
    })
    return b
}

func (b *NotificationServiceBuilder) WithRetry(maxRetries int) *NotificationServiceBuilder {
    b.decorators = append(b.decorators, DecoratorConfig{
        Type:   "retry",
        Params: map[string]interface{}{"maxRetries": maxRetries},
    })
    return b
}

func (b *NotificationServiceBuilder) Build() NotificationService {
    service := b.baseService
    factory := &NotificationDecoratorFactory{}

    for _, config := range b.decorators {
        service = factory.CreateDecorator(service, config)
    }

    return service
}
```

### 3. Configuration-driven Decoration

```go
type ServiceConfig struct {
    Decorators []DecoratorSpec `yaml:"decorators"`
}

type DecoratorSpec struct {
    Name   string                 `yaml:"name"`
    Config map[string]interface{} `yaml:"config"`
}

func BuildServiceFromConfig(baseService NotificationService, config ServiceConfig) NotificationService {
    service := baseService

    for _, spec := range config.Decorators {
        service = applyDecorator(service, spec)
    }

    return service
}

func applyDecorator(service NotificationService, spec DecoratorSpec) NotificationService {
    switch spec.Name {
    case "logging":
        logger, _ := zap.NewDevelopment()
        return NewLoggingNotificationDecorator(service, logger)
    case "retry":
        maxRetries := int(spec.Config["maxRetries"].(float64))
        return NewRetryNotificationDecorator(service, maxRetries, &ExponentialBackoff{})
    default:
        return service
    }
}
```

## Common Interview Questions

### 1. **How does Decorator pattern differ from Inheritance?**

**Answer:**
Decorator pattern provides composition-based extension instead of inheritance-based extension:

**Inheritance:**

```go
// Base class
type NotificationService struct{}

func (n *NotificationService) Send(message string) error {
    return sendNotification(message)
}

// Inheritance - creates class explosion
type LoggingNotificationService struct {
    NotificationService
    logger Logger
}

type RetryNotificationService struct {
    NotificationService
    maxRetries int
}

// Problem: What if you want both logging AND retry?
type LoggingRetryNotificationService struct {
    NotificationService
    logger     Logger
    maxRetries int
}
```

**Decorator:**

```go
// Component interface
type NotificationService interface {
    Send(message string) error
}

// Base implementation
type BasicNotificationService struct{}

func (b *BasicNotificationService) Send(message string) error {
    return sendNotification(message)
}

// Decorators
type LoggingDecorator struct {
    NotificationService
    logger Logger
}

type RetryDecorator struct {
    NotificationService
    maxRetries int
}

// Flexible composition
service := &BasicNotificationService{}
service = &LoggingDecorator{service, logger}
service = &RetryDecorator{service, 3}
```

**Key Differences:**

| Aspect           | Inheritance           | Decorator              |
| ---------------- | --------------------- | ---------------------- |
| **Flexibility**  | Static (compile-time) | Dynamic (runtime)      |
| **Combinations** | Class explosion       | Flexible composition   |
| **Dependencies** | Tight coupling        | Loose coupling         |
| **Reusability**  | Limited               | High                   |
| **Complexity**   | Hierarchy complexity  | Composition complexity |

### 2. **How do you handle decorator ordering and dependencies?**

**Answer:**
Decorator ordering matters and should be managed carefully:

**Order-dependent Decorators:**

```go
// Order matters: Logging should see the final result
service := baseService
service = NewRetryDecorator(service, 3)           // Retry first
service = NewRateLimitingDecorator(service, limiter) // Then rate limit
service = NewLoggingDecorator(service, logger)    // Finally log

// Wrong order might log retry attempts instead of final result
```

**Dependency Management:**

```go
type DecoratorChain struct {
    decorators []DecoratorFactory
    order      []string
}

func (d *DecoratorChain) AddDecorator(name string, factory DecoratorFactory) {
    d.decorators = append(d.decorators, factory)
    d.order = append(d.order, name)
}

func (d *DecoratorChain) Build(base NotificationService) NotificationService {
    service := base

    // Apply decorators in defined order
    for _, factory := range d.decorators {
        service = factory.Create(service)
    }

    return service
}

// Define standard order
chain := &DecoratorChain{}
chain.AddDecorator("validation", validationFactory)
chain.AddDecorator("authentication", authFactory)
chain.AddDecorator("rateLimit", rateLimitFactory)
chain.AddDecorator("retry", retryFactory)
chain.AddDecorator("logging", loggingFactory)
```

**Dependency Declaration:**

```go
type DecoratorMetadata struct {
    Name         string
    Dependencies []string
    Conflicts    []string
}

func (d *DecoratorChain) AddDecoratorWithMetadata(meta DecoratorMetadata, factory DecoratorFactory) error {
    // Check dependencies
    for _, dep := range meta.Dependencies {
        if !d.hasDecorator(dep) {
            return fmt.Errorf("missing dependency: %s", dep)
        }
    }

    // Check conflicts
    for _, conflict := range meta.Conflicts {
        if d.hasDecorator(conflict) {
            return fmt.Errorf("conflicting decorator: %s", conflict)
        }
    }

    d.decorators = append(d.decorators, factory)
    return nil
}
```

### 3. **How do you implement decorator pattern with error handling?**

**Answer:**
Error handling in decorators requires careful consideration of error propagation and recovery:

**Error Propagation:**

```go
type ErrorHandlingDecorator struct {
    NotificationService
    errorHandler func(error) error
}

func (e *ErrorHandlingDecorator) Send(message string) error {
    err := e.NotificationService.Send(message)
    if err != nil {
        return e.errorHandler(err)
    }
    return nil
}

// Different error handling strategies
func RetryErrorHandler(maxRetries int) func(error) error {
    return func(err error) error {
        // Retry logic
        for i := 0; i < maxRetries; i++ {
            // Retry the operation
            if retryErr := retry(); retryErr == nil {
                return nil
            }
        }
        return err
    }
}

func LogAndIgnoreErrorHandler(logger Logger) func(error) error {
    return func(err error) error {
        logger.Error("Operation failed", err)
        return nil // Ignore error
    }
}
```

**Circuit Breaker Integration:**

```go
type CircuitBreakerDecorator struct {
    NotificationService
    breaker *CircuitBreaker
}

func (c *CircuitBreakerDecorator) Send(message string) error {
    return c.breaker.Execute(func() error {
        return c.NotificationService.Send(message)
    })
}

type CircuitBreaker struct {
    state        CircuitState
    failureCount int
    threshold    int
    timeout      time.Duration
    lastFailure  time.Time
}

func (c *CircuitBreaker) Execute(fn func() error) error {
    switch c.state {
    case CircuitOpen:
        if time.Since(c.lastFailure) > c.timeout {
            c.state = CircuitHalfOpen
        } else {
            return fmt.Errorf("circuit breaker is open")
        }
    }

    err := fn()

    if err != nil {
        c.recordFailure()
    } else {
        c.recordSuccess()
    }

    return err
}
```

**Error Context Preservation:**

```go
type ErrorContext struct {
    Operation string
    Layer     string
    Metadata  map[string]interface{}
    Cause     error
}

func (e *ErrorContext) Error() string {
    return fmt.Sprintf("%s failed at %s: %v", e.Operation, e.Layer, e.Cause)
}

type ContextualErrorDecorator struct {
    NotificationService
    layer string
}

func (c *ContextualErrorDecorator) Send(message string) error {
    err := c.NotificationService.Send(message)
    if err != nil {
        return &ErrorContext{
            Operation: "Send",
            Layer:     c.layer,
            Metadata:  map[string]interface{}{"message": message},
            Cause:     err,
        }
    }
    return nil
}
```

### 4. **How do you test decorated objects effectively?**

**Answer:**
Testing decorators requires both unit testing of individual decorators and integration testing of decorator chains:

**Unit Testing Individual Decorators:**

```go
func TestLoggingDecorator(t *testing.T) {
    // Create mock service
    mockService := &MockNotificationService{}

    // Create logger with buffer
    var buf bytes.Buffer
    logger := log.New(&buf, "", 0)

    // Create decorator
    decorator := NewLoggingDecorator(mockService, logger)

    // Test success case
    mockService.On("Send", "test").Return(nil)
    err := decorator.Send("test")

    assert.NoError(t, err)
    assert.Contains(t, buf.String(), "Sending notification")
    assert.Contains(t, buf.String(), "test")

    // Test error case
    buf.Reset()
    mockService.On("Send", "error").Return(fmt.Errorf("send failed"))
    err = decorator.Send("error")

    assert.Error(t, err)
    assert.Contains(t, buf.String(), "Failed to send")
}
```

**Testing Decorator Chains:**

```go
func TestDecoratorChain(t *testing.T) {
    // Create test service
    mockService := &MockNotificationService{}

    // Create decorator chain
    service := NotificationService(mockService)
    service = NewRetryDecorator(service, 2)
    service = NewLoggingDecorator(service, logger)

    // Test that retry works
    mockService.On("Send", "retry-test").
        Return(fmt.Errorf("temp failure")).Once()
    mockService.On("Send", "retry-test").
        Return(nil).Once()

    err := service.Send("retry-test")
    assert.NoError(t, err)

    // Verify both calls were made
    mockService.AssertNumberOfCalls(t, "Send", 2)
}
```

**Mock-based Testing:**

```go
type MockNotificationService struct {
    mock.Mock
}

func (m *MockNotificationService) Send(message string) error {
    args := m.Called(message)
    return args.Error(0)
}

// Test decorator behavior without dependencies
func TestRetryDecorator(t *testing.T) {
    mockService := &MockNotificationService{}
    retryDecorator := NewRetryDecorator(mockService, 3)

    // Fail twice, succeed on third attempt
    mockService.On("Send", "test").Return(fmt.Errorf("failure")).Twice()
    mockService.On("Send", "test").Return(nil).Once()

    err := retryDecorator.Send("test")
    assert.NoError(t, err)
    mockService.AssertNumberOfCalls(t, "Send", 3)
}
```

**Integration Testing:**

```go
func TestFullNotificationPipeline(t *testing.T) {
    // Setup real dependencies
    emailSender := &TestEmailSender{}
    smsSender := &TestSMSSender{}

    // Create full pipeline
    service := BuildNotificationService(emailSender, smsSender)

    // Test end-to-end functionality
    err := service.Send("integration-test")
    assert.NoError(t, err)

    // Verify side effects
    assert.Equal(t, 1, emailSender.CallCount)
    assert.True(t, logging.ContainsLog("Successfully sent"))
}
```

### 5. **How do you handle performance in deeply nested decorators?**

**Answer:**
Performance optimization in decorator chains involves several strategies:

**Lazy Evaluation:**

```go
type LazyDecorator struct {
    NotificationService
    operation func() error
    result    error
    executed  bool
    mu        sync.Once
}

func (l *LazyDecorator) Send(message string) error {
    l.mu.Do(func() {
        l.result = l.NotificationService.Send(message)
        l.executed = true
    })
    return l.result
}
```

**Caching Decorator:**

```go
type CachingDecorator struct {
    NotificationService
    cache map[string]error
    ttl   time.Duration
    mu    sync.RWMutex
}

func (c *CachingDecorator) Send(message string) error {
    c.mu.RLock()
    if cached, exists := c.cache[message]; exists {
        c.mu.RUnlock()
        return cached
    }
    c.mu.RUnlock()

    err := c.NotificationService.Send(message)

    c.mu.Lock()
    c.cache[message] = err
    c.mu.Unlock()

    return err
}
```

**Async Processing:**

```go
type AsyncDecorator struct {
    NotificationService
    workers   int
    queue     chan AsyncTask
    wg        sync.WaitGroup
}

type AsyncTask struct {
    Message  string
    Response chan error
}

func (a *AsyncDecorator) Send(message string) error {
    task := AsyncTask{
        Message:  message,
        Response: make(chan error, 1),
    }

    a.queue <- task
    return <-task.Response
}

func (a *AsyncDecorator) worker() {
    defer a.wg.Done()

    for task := range a.queue {
        err := a.NotificationService.Send(task.Message)
        task.Response <- err
    }
}
```

**Profiling and Benchmarking:**

```go
func BenchmarkDecoratorChain(b *testing.B) {
    service := buildTestService()

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        service.Send("benchmark-message")
    }
}

func BenchmarkNestedDecorators(b *testing.B) {
    services := []NotificationService{
        buildServiceWith1Decorator(),
        buildServiceWith5Decorators(),
        buildServiceWith10Decorators(),
    }

    for i, service := range services {
        b.Run(fmt.Sprintf("decorators-%d", (i+1)*5-4), func(b *testing.B) {
            for j := 0; j < b.N; j++ {
                service.Send("test")
            }
        })
    }
}
```
