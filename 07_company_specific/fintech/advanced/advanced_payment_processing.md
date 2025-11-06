---
# Auto-generated front matter
Title: Advanced Payment Processing
LastUpdated: 2025-11-06T20:45:58.482334
Tags: []
Status: draft
---

# Advanced Payment Processing Systems

Advanced payment processing architectures and patterns for fintech companies.

## 🎯 Payment Processing Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Payment API   │    │  Payment Engine │    │  Settlement     │
│   Gateway       │────│  (Core Logic)   │────│  Engine         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  Risk & Fraud   │
                       │  Management     │
                       └─────────────────┘
```

### Payment Flow Components
- **Payment Gateway**: Entry point for payment requests
- **Payment Engine**: Core business logic and orchestration
- **Risk Engine**: Fraud detection and risk assessment
- **Settlement Engine**: Funds transfer and reconciliation
- **Notification Service**: Real-time status updates

## 🔧 Core Payment Engine

### Payment Processing Service
```go
type PaymentProcessor struct {
    gateway        PaymentGateway
    riskEngine     RiskEngine
    settlement     SettlementEngine
    notification   NotificationService
    repository     PaymentRepository
    cache          Cache
}

type PaymentRequest struct {
    ID              string            `json:"id"`
    Amount          decimal.Decimal   `json:"amount"`
    Currency        string            `json:"currency"`
    PaymentMethod   PaymentMethod     `json:"payment_method"`
    CustomerID      string            `json:"customer_id"`
    MerchantID      string            `json:"merchant_id"`
    Metadata        map[string]string `json:"metadata"`
    CallbackURL     string            `json:"callback_url"`
    WebhookURL      string            `json:"webhook_url"`
}

type PaymentResponse struct {
    ID              string            `json:"id"`
    Status          PaymentStatus     `json:"status"`
    TransactionID   string            `json:"transaction_id"`
    GatewayResponse interface{}       `json:"gateway_response"`
    RiskScore       float64           `json:"risk_score"`
    ProcessedAt     time.Time         `json:"processed_at"`
}

func (pp *PaymentProcessor) ProcessPayment(req PaymentRequest) (*PaymentResponse, error) {
    // Generate payment ID
    paymentID := generatePaymentID()
    
    // Risk assessment
    riskScore, riskFactors, err := pp.riskEngine.AssessRisk(req)
    if err != nil {
        return nil, err
    }
    
    // Check risk threshold
    if riskScore > pp.riskEngine.GetThreshold() {
        return &PaymentResponse{
            ID:        paymentID,
            Status:    PaymentRejected,
            RiskScore: riskScore,
        }, nil
    }
    
    // Process payment through gateway
    gatewayResp, err := pp.gateway.ProcessPayment(req)
    if err != nil {
        return nil, err
    }
    
    // Create payment record
    payment := &Payment{
        ID:              paymentID,
        Amount:          req.Amount,
        Currency:        req.Currency,
        Status:          PaymentProcessing,
        GatewayResponse: gatewayResp,
        RiskScore:       riskScore,
        RiskFactors:     riskFactors,
        CreatedAt:       time.Now(),
    }
    
    // Save payment
    if err := pp.repository.Save(payment); err != nil {
        return nil, err
    }
    
    // Process settlement asynchronously
    go pp.processSettlement(payment)
    
    // Send notification
    go pp.notification.SendPaymentStatus(payment)
    
    return &PaymentResponse{
        ID:              paymentID,
        Status:          payment.Status,
        TransactionID:   gatewayResp.TransactionID,
        GatewayResponse: gatewayResp,
        RiskScore:       riskScore,
        ProcessedAt:     payment.CreatedAt,
    }, nil
}
```

### Payment Gateway Interface
```go
type PaymentGateway interface {
    ProcessPayment(req PaymentRequest) (*GatewayResponse, error)
    RefundPayment(transactionID string, amount decimal.Decimal) (*GatewayResponse, error)
    CapturePayment(transactionID string, amount decimal.Decimal) (*GatewayResponse, error)
    VoidPayment(transactionID string) (*GatewayResponse, error)
    GetPaymentStatus(transactionID string) (*PaymentStatus, error)
}

type GatewayResponse struct {
    TransactionID   string                 `json:"transaction_id"`
    Status          string                 `json:"status"`
    GatewayID       string                 `json:"gateway_id"`
    ResponseCode    string                 `json:"response_code"`
    ResponseMessage string                 `json:"response_message"`
    RawResponse     map[string]interface{} `json:"raw_response"`
    ProcessedAt     time.Time              `json:"processed_at"`
}

// Razorpay Gateway Implementation
type RazorpayGateway struct {
    client    *razorpay.Client
    webhook   WebhookHandler
}

func (rg *RazorpayGateway) ProcessPayment(req PaymentRequest) (*GatewayResponse, error) {
    // Create Razorpay order
    orderReq := &razorpay.OrderCreateParams{
        Amount:   int(req.Amount.Mul(decimal.NewFromInt(100)).IntPart()), // Convert to paise
        Currency: req.Currency,
        Receipt:  req.ID,
    }
    
    order, err := rg.client.Order.Create(orderReq, nil)
    if err != nil {
        return nil, err
    }
    
    // Create payment
    paymentReq := &razorpay.PaymentCreateParams{
        Amount:   order.Amount,
        Currency: order.Currency,
        OrderID:  order.ID,
    }
    
    payment, err := rg.client.Payment.Create(paymentReq, nil)
    if err != nil {
        return nil, err
    }
    
    return &GatewayResponse{
        TransactionID:   payment.ID,
        Status:          payment.Status,
        GatewayID:       "razorpay",
        ResponseCode:    payment.Status,
        ResponseMessage: "Payment processed successfully",
        RawResponse:     payment,
        ProcessedAt:     time.Now(),
    }, nil
}
```

## 🛡️ Risk and Fraud Management

### Risk Engine
```go
type RiskEngine struct {
    rules        []RiskRule
    mlModel      MLModel
    thresholds   RiskThresholds
    cache        Cache
}

type RiskRule struct {
    ID          string
    Name        string
    Condition   func(PaymentRequest) bool
    RiskScore   float64
    Description string
}

type RiskThresholds struct {
    Low    float64
    Medium float64
    High   float64
    Block  float64
}

func (re *RiskEngine) AssessRisk(req PaymentRequest) (float64, []string, error) {
    var totalScore float64
    var riskFactors []string
    
    // Check cached risk score
    cacheKey := fmt.Sprintf("risk:%s:%s", req.CustomerID, req.Amount.String())
    if cached, found := re.cache.Get(cacheKey); found {
        return cached.(float64), []string{}, nil
    }
    
    // Apply rule-based checks
    for _, rule := range re.rules {
        if rule.Condition(req) {
            totalScore += rule.RiskScore
            riskFactors = append(riskFactors, rule.Description)
        }
    }
    
    // Apply ML model
    mlScore, err := re.mlModel.Predict(req)
    if err == nil {
        totalScore += mlScore
        riskFactors = append(riskFactors, "ML Risk Detection")
    }
    
    // Cache result
    re.cache.Set(cacheKey, totalScore, 5*time.Minute)
    
    return totalScore, riskFactors, nil
}

func (re *RiskEngine) GetThreshold() float64 {
    return re.thresholds.Block
}

// Risk Rules
var DefaultRiskRules = []RiskRule{
    {
        ID:   "high_amount",
        Name: "High Amount Check",
        Condition: func(req PaymentRequest) bool {
            return req.Amount.GreaterThan(decimal.NewFromInt(100000)) // > 1 lakh
        },
        RiskScore:   0.3,
        Description: "High transaction amount",
    },
    {
        ID:   "new_customer",
        Name: "New Customer Check",
        Condition: func(req PaymentRequest) bool {
            // Check if customer is new (within 30 days)
            return isNewCustomer(req.CustomerID)
        },
        RiskScore:   0.2,
        Description: "New customer",
    },
    {
        ID:   "velocity_check",
        Name: "Velocity Check",
        Condition: func(req PaymentRequest) bool {
            // Check transaction velocity
            return getTransactionVelocity(req.CustomerID) > 10 // > 10 transactions per hour
        },
        RiskScore:   0.4,
        Description: "High transaction velocity",
    },
}
```

### Fraud Detection
```go
type FraudDetector struct {
    mlModel      MLModel
    rules        []FraudRule
    blacklist    BlacklistService
    whitelist    WhitelistService
}

type FraudRule struct {
    ID          string
    Name        string
    Condition   func(PaymentRequest) bool
    Action      FraudAction
    Description string
}

type FraudAction string

const (
    FraudActionBlock    FraudAction = "block"
    FraudActionReview   FraudAction = "review"
    FraudActionAllow    FraudAction = "allow"
)

func (fd *FraudDetector) DetectFraud(req PaymentRequest) (FraudAction, []string, error) {
    var reasons []string
    
    // Check blacklist
    if fd.blacklist.IsBlacklisted(req.CustomerID, req.PaymentMethod) {
        return FraudActionBlock, []string{"Customer or payment method blacklisted"}, nil
    }
    
    // Check whitelist
    if fd.whitelist.IsWhitelisted(req.CustomerID) {
        return FraudActionAllow, []string{"Customer whitelisted"}, nil
    }
    
    // Apply fraud rules
    for _, rule := range fd.rules {
        if rule.Condition(req) {
            reasons = append(reasons, rule.Description)
            if rule.Action == FraudActionBlock {
                return rule.Action, reasons, nil
            }
        }
    }
    
    // Apply ML model
    mlScore, err := fd.mlModel.Predict(req)
    if err == nil && mlScore > 0.8 {
        reasons = append(reasons, "ML fraud detection")
        return FraudActionBlock, reasons, nil
    }
    
    if len(reasons) > 0 {
        return FraudActionReview, reasons, nil
    }
    
    return FraudActionAllow, []string{}, nil
}
```

## 💰 Settlement Engine

### Settlement Processing
```go
type SettlementEngine struct {
    bankAPI       BankAPI
    ledger        LedgerService
    notification  NotificationService
    repository    SettlementRepository
}

type Settlement struct {
    ID              string          `json:"id"`
    PaymentID       string          `json:"payment_id"`
    MerchantID      string          `json:"merchant_id"`
    Amount          decimal.Decimal `json:"amount"`
    Currency        string          `json:"currency"`
    Status          SettlementStatus `json:"status"`
    SettlementDate  time.Time       `json:"settlement_date"`
    BankReference   string          `json:"bank_reference"`
    CreatedAt       time.Time       `json:"created_at"`
    UpdatedAt       time.Time       `json:"updated_at"`
}

type SettlementStatus string

const (
    SettlementPending    SettlementStatus = "pending"
    SettlementProcessing SettlementStatus = "processing"
    SettlementCompleted  SettlementStatus = "completed"
    SettlementFailed     SettlementStatus = "failed"
)

func (se *SettlementEngine) ProcessSettlement(payment *Payment) error {
    // Create settlement record
    settlement := &Settlement{
        ID:         generateSettlementID(),
        PaymentID:  payment.ID,
        MerchantID: payment.MerchantID,
        Amount:     payment.Amount,
        Currency:   payment.Currency,
        Status:     SettlementPending,
        CreatedAt:  time.Now(),
    }
    
    // Save settlement
    if err := se.repository.Save(settlement); err != nil {
        return err
    }
    
    // Process settlement
    go se.processSettlementAsync(settlement)
    
    return nil
}

func (se *SettlementEngine) processSettlementAsync(settlement *Settlement) {
    // Update status to processing
    settlement.Status = SettlementProcessing
    se.repository.Update(settlement)
    
    // Process bank transfer
    bankResp, err := se.bankAPI.Transfer(settlement.MerchantID, settlement.Amount, settlement.Currency)
    if err != nil {
        settlement.Status = SettlementFailed
        se.repository.Update(settlement)
        return
    }
    
    // Update settlement with bank reference
    settlement.BankReference = bankResp.Reference
    settlement.SettlementDate = time.Now()
    settlement.Status = SettlementCompleted
    se.repository.Update(settlement)
    
    // Update ledger
    se.ledger.RecordSettlement(settlement)
    
    // Send notification
    se.notification.SendSettlementNotification(settlement)
}
```

## 🔄 Payment Status Management

### Status Tracking
```go
type PaymentStatusManager struct {
    repository  PaymentRepository
    gateway     PaymentGateway
    notification NotificationService
    cache       Cache
}

func (psm *PaymentStatusManager) UpdatePaymentStatus(paymentID string) error {
    // Get payment from repository
    payment, err := psm.repository.GetByID(paymentID)
    if err != nil {
        return err
    }
    
    // Get status from gateway
    gatewayStatus, err := psm.gateway.GetPaymentStatus(payment.TransactionID)
    if err != nil {
        return err
    }
    
    // Update payment status
    oldStatus := payment.Status
    payment.Status = mapGatewayStatus(gatewayStatus.Status)
    payment.UpdatedAt = time.Now()
    
    // Save updated payment
    if err := psm.repository.Update(payment); err != nil {
        return err
    }
    
    // Send notification if status changed
    if oldStatus != payment.Status {
        go psm.notification.SendPaymentStatusUpdate(payment)
    }
    
    // Update cache
    psm.cache.Set(fmt.Sprintf("payment:%s", paymentID), payment, 1*time.Hour)
    
    return nil
}

func mapGatewayStatus(gatewayStatus string) PaymentStatus {
    switch gatewayStatus {
    case "authorized":
        return PaymentAuthorized
    case "captured":
        return PaymentCompleted
    case "refunded":
        return PaymentRefunded
    case "failed":
        return PaymentFailed
    default:
        return PaymentProcessing
    }
}
```

## 📊 Payment Analytics

### Analytics Service
```go
type PaymentAnalytics struct {
    repository PaymentRepository
    cache      Cache
    metrics    MetricsService
}

type PaymentMetrics struct {
    TotalTransactions    int64           `json:"total_transactions"`
    TotalAmount          decimal.Decimal `json:"total_amount"`
    SuccessRate          float64         `json:"success_rate"`
    AverageAmount        decimal.Decimal `json:"average_amount"`
    TopPaymentMethods    []PaymentMethodStats `json:"top_payment_methods"`
    HourlyDistribution   []HourlyStats   `json:"hourly_distribution"`
}

type PaymentMethodStats struct {
    Method string          `json:"method"`
    Count  int64           `json:"count"`
    Amount decimal.Decimal `json:"amount"`
}

type HourlyStats struct {
    Hour  int             `json:"hour"`
    Count int64           `json:"count"`
    Amount decimal.Decimal `json:"amount"`
}

func (pa *PaymentAnalytics) GetMetrics(merchantID string, startDate, endDate time.Time) (*PaymentMetrics, error) {
    // Check cache first
    cacheKey := fmt.Sprintf("metrics:%s:%s:%s", merchantID, startDate.Format("2006-01-02"), endDate.Format("2006-01-02"))
    if cached, found := pa.cache.Get(cacheKey); found {
        return cached.(*PaymentMetrics), nil
    }
    
    // Get payments from repository
    payments, err := pa.repository.GetByMerchantAndDateRange(merchantID, startDate, endDate)
    if err != nil {
        return nil, err
    }
    
    // Calculate metrics
    metrics := pa.calculateMetrics(payments)
    
    // Cache result
    pa.cache.Set(cacheKey, metrics, 5*time.Minute)
    
    return metrics, nil
}

func (pa *PaymentAnalytics) calculateMetrics(payments []Payment) *PaymentMetrics {
    var totalTransactions int64
    var totalAmount decimal.Decimal
    var successfulTransactions int64
    
    paymentMethodStats := make(map[string]*PaymentMethodStats)
    hourlyStats := make(map[int]*HourlyStats)
    
    for _, payment := range payments {
        totalTransactions++
        totalAmount = totalAmount.Add(payment.Amount)
        
        if payment.Status == PaymentCompleted {
            successfulTransactions++
        }
        
        // Payment method stats
        method := string(payment.PaymentMethod.Type)
        if stats, exists := paymentMethodStats[method]; exists {
            stats.Count++
            stats.Amount = stats.Amount.Add(payment.Amount)
        } else {
            paymentMethodStats[method] = &PaymentMethodStats{
                Method: method,
                Count:  1,
                Amount: payment.Amount,
            }
        }
        
        // Hourly stats
        hour := payment.CreatedAt.Hour()
        if stats, exists := hourlyStats[hour]; exists {
            stats.Count++
            stats.Amount = stats.Amount.Add(payment.Amount)
        } else {
            hourlyStats[hour] = &HourlyStats{
                Hour:   hour,
                Count:  1,
                Amount: payment.Amount,
            }
        }
    }
    
    // Calculate success rate
    successRate := float64(0)
    if totalTransactions > 0 {
        successRate = float64(successfulTransactions) / float64(totalTransactions) * 100
    }
    
    // Calculate average amount
    averageAmount := decimal.Zero
    if totalTransactions > 0 {
        averageAmount = totalAmount.Div(decimal.NewFromInt(totalTransactions))
    }
    
    // Convert maps to slices
    topPaymentMethods := make([]PaymentMethodStats, 0, len(paymentMethodStats))
    for _, stats := range paymentMethodStats {
        topPaymentMethods = append(topPaymentMethods, *stats)
    }
    
    hourlyDistribution := make([]HourlyStats, 0, len(hourlyStats))
    for _, stats := range hourlyStats {
        hourlyDistribution = append(hourlyDistribution, *stats)
    }
    
    return &PaymentMetrics{
        TotalTransactions:    totalTransactions,
        TotalAmount:          totalAmount,
        SuccessRate:          successRate,
        AverageAmount:        averageAmount,
        TopPaymentMethods:    topPaymentMethods,
        HourlyDistribution:   hourlyDistribution,
    }
}
```

## 🔐 Security and Compliance

### PCI DSS Compliance
```go
type PCICompliance struct {
    encryption EncryptionService
    tokenization TokenizationService
    audit      AuditService
}

func (pci *PCICompliance) ProcessSensitiveData(data map[string]interface{}) (map[string]interface{}, error) {
    processed := make(map[string]interface{})
    
    for key, value := range data {
        if pci.isSensitiveField(key) {
            // Tokenize sensitive data
            token, err := pci.tokenization.Tokenize(value.(string))
            if err != nil {
                return nil, err
            }
            processed[key] = token
        } else {
            processed[key] = value
        }
    }
    
    // Audit the processing
    pci.audit.LogDataProcessing(data, processed)
    
    return processed, nil
}

func (pci *PCICompliance) isSensitiveField(field string) bool {
    sensitiveFields := []string{
        "card_number",
        "cvv",
        "expiry_date",
        "cardholder_name",
    }
    
    for _, sensitive := range sensitiveFields {
        if field == sensitive {
            return true
        }
    }
    
    return false
}
```

## 🎯 Best Practices

### Payment Processing
1. **Idempotency**: Ensure payment requests are idempotent
2. **Atomicity**: Use transactions for critical operations
3. **Monitoring**: Comprehensive monitoring and alerting
4. **Security**: Implement proper security measures
5. **Compliance**: Ensure regulatory compliance

### Error Handling
1. **Graceful Degradation**: Handle failures gracefully
2. **Retry Logic**: Implement exponential backoff
3. **Circuit Breaker**: Prevent cascade failures
4. **Logging**: Comprehensive error logging
5. **Alerting**: Real-time error alerting

### Performance
1. **Caching**: Cache frequently accessed data
2. **Async Processing**: Use async for non-critical operations
3. **Database Optimization**: Optimize database queries
4. **Load Balancing**: Distribute load across instances
5. **Monitoring**: Monitor performance metrics

---

**Last Updated**: December 2024  
**Category**: Advanced Payment Processing  
**Complexity**: Senior Level
