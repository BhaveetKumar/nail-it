---
# Auto-generated front matter
Title: Observer Pattern
LastUpdated: 2025-11-06T20:45:58.543791
Tags: []
Status: draft
---

# Observer Pattern

Comprehensive guide to the Observer pattern for Razorpay interviews.

## 🎯 Observer Pattern Overview

The Observer pattern defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

### Key Benefits
- **Loose Coupling**: Subject and observers are loosely coupled
- **Dynamic Relationships**: Can add/remove observers at runtime
- **Broadcast Communication**: One-to-many communication
- **Open/Closed Principle**: Easy to add new observers

## 🚀 Implementation Examples

### Basic Observer Pattern
```go
// Observer Interface
type Observer interface {
    Update(data interface{})
}

// Subject Interface
type Subject interface {
    Attach(observer Observer)
    Detach(observer Observer)
    Notify()
}

// Concrete Subject
type PaymentSubject struct {
    observers []Observer
    state     PaymentState
    mutex     sync.RWMutex
}

type PaymentState struct {
    PaymentID string
    Status    string
    Amount    int64
    Timestamp time.Time
}

func NewPaymentSubject() *PaymentSubject {
    return &PaymentSubject{
        observers: make([]Observer, 0),
    }
}

func (ps *PaymentSubject) Attach(observer Observer) {
    ps.mutex.Lock()
    defer ps.mutex.Unlock()
    
    ps.observers = append(ps.observers, observer)
}

func (ps *PaymentSubject) Detach(observer Observer) {
    ps.mutex.Lock()
    defer ps.mutex.Unlock()
    
    for i, obs := range ps.observers {
        if obs == observer {
            ps.observers = append(ps.observers[:i], ps.observers[i+1:]...)
            break
        }
    }
}

func (ps *PaymentSubject) Notify() {
    ps.mutex.RLock()
    observers := make([]Observer, len(ps.observers))
    copy(observers, ps.observers)
    ps.mutex.RUnlock()
    
    for _, observer := range observers {
        observer.Update(ps.state)
    }
}

func (ps *PaymentSubject) SetState(state PaymentState) {
    ps.mutex.Lock()
    ps.state = state
    ps.mutex.Unlock()
    
    ps.Notify()
}
```

### Concrete Observers
```go
// Email Notification Observer
type EmailNotificationObserver struct {
    emailService EmailService
}

func NewEmailNotificationObserver(emailService EmailService) *EmailNotificationObserver {
    return &EmailNotificationObserver{
        emailService: emailService,
    }
}

func (eno *EmailNotificationObserver) Update(data interface{}) {
    state, ok := data.(PaymentState)
    if !ok {
        return
    }
    
    // Send email notification based on payment status
    switch state.Status {
    case "success":
        eno.emailService.SendPaymentSuccessEmail(state.PaymentID, state.Amount)
    case "failed":
        eno.emailService.SendPaymentFailedEmail(state.PaymentID, state.Amount)
    case "refunded":
        eno.emailService.SendRefundEmail(state.PaymentID, state.Amount)
    }
}

// SMS Notification Observer
type SMSNotificationObserver struct {
    smsService SMSService
}

func NewSMSNotificationObserver(smsService SMSService) *SMSNotificationObserver {
    return &SMSNotificationObserver{
        smsService: smsService,
    }
}

func (sno *SMSNotificationObserver) Update(data interface{}) {
    state, ok := data.(PaymentState)
    if !ok {
        return
    }
    
    // Send SMS notification based on payment status
    switch state.Status {
    case "success":
        sno.smsService.SendPaymentSuccessSMS(state.PaymentID, state.Amount)
    case "failed":
        sno.smsService.SendPaymentFailedSMS(state.PaymentID, state.Amount)
    }
}

// Analytics Observer
type AnalyticsObserver struct {
    analyticsService AnalyticsService
}

func NewAnalyticsObserver(analyticsService AnalyticsService) *AnalyticsObserver {
    return &AnalyticsObserver{
        analyticsService: analyticsService,
    }
}

func (ao *AnalyticsObserver) Update(data interface{}) {
    state, ok := data.(PaymentState)
    if !ok {
        return
    }
    
    // Track payment event in analytics
    event := AnalyticsEvent{
        EventType: "payment_" + state.Status,
        PaymentID: state.PaymentID,
        Amount:    state.Amount,
        Timestamp: state.Timestamp,
    }
    
    ao.analyticsService.TrackEvent(event)
}
```

## 🔧 Advanced Observer Patterns

### Event-Driven Observer
```go
// Event Interface
type Event interface {
    GetType() string
    GetData() interface{}
    GetTimestamp() time.Time
}

// Payment Event
type PaymentEvent struct {
    Type      string
    Data      interface{}
    Timestamp time.Time
}

func (pe *PaymentEvent) GetType() string {
    return pe.Type
}

func (pe *PaymentEvent) GetData() interface{} {
    return pe.Data
}

func (pe *PaymentEvent) GetTimestamp() time.Time {
    return pe.Timestamp
}

// Event Observer Interface
type EventObserver interface {
    HandleEvent(event Event)
    GetEventTypes() []string
}

// Event Bus
type EventBus struct {
    observers map[string][]EventObserver
    mutex     sync.RWMutex
}

func NewEventBus() *EventBus {
    return &EventBus{
        observers: make(map[string][]EventObserver),
    }
}

func (eb *EventBus) Subscribe(eventType string, observer EventObserver) {
    eb.mutex.Lock()
    defer eb.mutex.Unlock()
    
    eb.observers[eventType] = append(eb.observers[eventType], observer)
}

func (eb *EventBus) Unsubscribe(eventType string, observer EventObserver) {
    eb.mutex.Lock()
    defer eb.mutex.Unlock()
    
    observers := eb.observers[eventType]
    for i, obs := range observers {
        if obs == observer {
            eb.observers[eventType] = append(observers[:i], observers[i+1:]...)
            break
        }
    }
}

func (eb *EventBus) Publish(event Event) {
    eb.mutex.RLock()
    observers := eb.observers[event.GetType()]
    eb.mutex.RUnlock()
    
    for _, observer := range observers {
        go observer.HandleEvent(event)
    }
}
```

### Filtered Observer
```go
// Filtered Observer
type FilteredObserver struct {
    observer    Observer
    filter      func(interface{}) bool
    eventTypes  []string
}

func NewFilteredObserver(observer Observer, filter func(interface{}) bool, eventTypes []string) *FilteredObserver {
    return &FilteredObserver{
        observer:   observer,
        filter:     filter,
        eventTypes: eventTypes,
    }
}

func (fo *FilteredObserver) Update(data interface{}) {
    // Check if data passes the filter
    if fo.filter != nil && !fo.filter(data) {
        return
    }
    
    // Check if event type is in the allowed list
    if len(fo.eventTypes) > 0 {
        if state, ok := data.(PaymentState); ok {
            allowed := false
            for _, eventType := range fo.eventTypes {
                if state.Status == eventType {
                    allowed = true
                    break
                }
            }
            if !allowed {
                return
            }
        }
    }
    
    fo.observer.Update(data)
}
```

## 🎯 Razorpay-Specific Examples

### Payment Processing Observer
```go
// Payment Processing Observer for Razorpay
type PaymentProcessingObserver struct {
    paymentService PaymentService
    webhookService WebhookService
}

func NewPaymentProcessingObserver(paymentService PaymentService, webhookService WebhookService) *PaymentProcessingObserver {
    return &PaymentProcessingObserver{
        paymentService: paymentService,
        webhookService: webhookService,
    }
}

func (ppo *PaymentProcessingObserver) Update(data interface{}) {
    state, ok := data.(PaymentState)
    if !ok {
        return
    }
    
    // Process payment based on status
    switch state.Status {
    case "authorized":
        ppo.handleAuthorizedPayment(state)
    case "captured":
        ppo.handleCapturedPayment(state)
    case "failed":
        ppo.handleFailedPayment(state)
    case "refunded":
        ppo.handleRefundedPayment(state)
    }
}

func (ppo *PaymentProcessingObserver) handleAuthorizedPayment(state PaymentState) {
    // Auto-capture if configured
    if ppo.paymentService.ShouldAutoCapture(state.PaymentID) {
        go ppo.paymentService.CapturePayment(state.PaymentID)
    }
    
    // Send webhook notification
    ppo.webhookService.SendWebhook(state.PaymentID, "payment.authorized", state)
}

func (ppo *PaymentProcessingObserver) handleCapturedPayment(state PaymentState) {
    // Update merchant balance
    ppo.paymentService.UpdateMerchantBalance(state.PaymentID, state.Amount)
    
    // Send webhook notification
    ppo.webhookService.SendWebhook(state.PaymentID, "payment.captured", state)
}

func (ppo *PaymentProcessingObserver) handleFailedPayment(state PaymentState) {
    // Log failure reason
    ppo.paymentService.LogPaymentFailure(state.PaymentID, "Payment failed")
    
    // Send webhook notification
    ppo.webhookService.SendWebhook(state.PaymentID, "payment.failed", state)
}

func (ppo *PaymentProcessingObserver) handleRefundedPayment(state PaymentState) {
    // Update merchant balance
    ppo.paymentService.UpdateMerchantBalance(state.PaymentID, -state.Amount)
    
    // Send webhook notification
    ppo.webhookService.SendWebhook(state.PaymentID, "payment.refunded", state)
}
```

### Fraud Detection Observer
```go
// Fraud Detection Observer
type FraudDetectionObserver struct {
    fraudService FraudDetectionService
    riskScore    int
}

func NewFraudDetectionObserver(fraudService FraudDetectionService, riskScore int) *FraudDetectionObserver {
    return &FraudDetectionObserver{
        fraudService: fraudService,
        riskScore:    riskScore,
    }
}

func (fdo *FraudDetectionObserver) Update(data interface{}) {
    state, ok := data.(PaymentState)
    if !ok {
        return
    }
    
    // Only process high-risk payments
    if state.Amount < int64(fdo.riskScore*100) {
        return
    }
    
    // Perform fraud detection
    go func() {
        isFraudulent, reason := fdo.fraudService.DetectFraud(state.PaymentID, state.Amount)
        
        if isFraudulent {
            fdo.fraudService.BlockPayment(state.PaymentID, reason)
        }
    }()
}
```

## 🎯 Best Practices

### Design Principles
1. **Single Responsibility**: Each observer should have one responsibility
2. **Open/Closed**: Easy to add new observers without modifying existing code
3. **Loose Coupling**: Subject and observers should be loosely coupled
4. **Interface Segregation**: Use specific interfaces for different types of observers

### Implementation Guidelines
1. **Thread Safety**: Use proper synchronization for concurrent access
2. **Error Handling**: Handle errors gracefully in observers
3. **Performance**: Consider using goroutines for expensive operations
4. **Memory Management**: Properly clean up observers to prevent memory leaks

### Common Pitfalls
1. **Memory Leaks**: Not removing observers can cause memory leaks
2. **Infinite Loops**: Observers should not trigger the same event
3. **Performance Issues**: Too many observers can impact performance
4. **Error Propagation**: Errors in observers should not crash the system

---

**Last Updated**: December 2024  
**Category**: Observer Pattern  
**Complexity**: Intermediate Level
