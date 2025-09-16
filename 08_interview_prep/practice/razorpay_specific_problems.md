# Razorpay-Specific Interview Problems

## Table of Contents
- [Introduction](#introduction/)
- [Payment Processing Problems](#payment-processing-problems/)
- [Financial Systems Design](#financial-systems-design/)
- [Compliance and Security](#compliance-and-security/)
- [Scalability Challenges](#scalability-challenges/)
- [Real-Time Systems](#real-time-systems/)
- [Data Engineering](#data-engineering/)
- [Machine Learning Applications](#machine-learning-applications/)

## Introduction

Razorpay-specific interview problems focus on the unique challenges faced in building and scaling financial technology platforms. These problems test your understanding of payment processing, financial regulations, high-availability systems, and fintech-specific technical challenges.

## Payment Processing Problems

### Problem 1: Design a Payment Gateway

**Problem Statement**: Design a payment gateway that can handle 1 million transactions per day with 99.99% uptime and sub-second response times.

**Requirements**:
- Support multiple payment methods (cards, UPI, net banking, wallets)
- Handle payment failures and retries
- Implement idempotency
- Support webhooks and notifications
- Ensure PCI DSS compliance
- Handle currency conversion
- Support refunds and chargebacks

**Solution Framework**:

```go
// Payment Gateway Core Components
type PaymentGateway struct {
    processors    map[string]*PaymentProcessor
    router        *PaymentRouter
    validator     *PaymentValidator
    encryptor     *PaymentEncryptor
    webhookMgr    *WebhookManager
    auditLogger   *AuditLogger
    rateLimiter   *RateLimiter
}

type PaymentRequest struct {
    ID              string
    Amount          int64
    Currency        string
    PaymentMethod   string
    CustomerID      string
    MerchantID      string
    OrderID         string
    CallbackURL     string
    Metadata        map[string]string
    Timestamp       time.Time
}

type PaymentResponse struct {
    ID              string
    Status          string
    TransactionID   string
    GatewayRefID    string
    Amount          int64
    Currency        string
    FailureReason   string
    Timestamp       time.Time
}

// Key Design Considerations:
// 1. Microservices Architecture
// 2. Event-Driven Design
// 3. Circuit Breaker Pattern
// 4. Database Sharding
// 5. Caching Strategy
// 6. Monitoring and Alerting
```

### Problem 2: Implement Payment Routing

**Problem Statement**: Design a payment routing system that automatically selects the best payment processor based on success rates, costs, and processing times.

**Requirements**:
- Dynamic routing based on real-time metrics
- Failover mechanisms
- Cost optimization
- A/B testing support
- Real-time monitoring

**Solution**:

```go
// Payment Router Implementation
type PaymentRouter struct {
    processors    map[string]*PaymentProcessor
    metrics       *RoutingMetrics
    rules         []*RoutingRule
    fallbackChain []string
    mu            sync.RWMutex
}

type RoutingRule struct {
    Condition string
    Processor string
    Priority  int
    Weight    float64
}

type RoutingMetrics struct {
    successRates  map[string]float64
    avgLatency    map[string]time.Duration
    costs         map[string]float64
    lastUpdated   time.Time
    mu            sync.RWMutex
}

func (pr *PaymentRouter) RoutePayment(request *PaymentRequest) (*PaymentProcessor, error) {
    // Get current metrics
    metrics := pr.metrics.GetCurrentMetrics()
    
    // Apply routing rules
    for _, rule := range pr.rules {
        if pr.evaluateRule(rule, request, metrics) {
            processor := pr.processors[rule.Processor]
            if processor.IsHealthy() {
                return processor, nil
            }
        }
    }
    
    // Fallback to best available processor
    return pr.selectBestProcessor(request, metrics)
}

func (pr *PaymentRouter) selectBestProcessor(request *PaymentRequest, metrics *RoutingMetrics) (*PaymentProcessor, error) {
    bestScore := 0.0
    bestProcessor := ""
    
    for name, processor := range pr.processors {
        if !processor.IsHealthy() {
            continue
        }
        
        score := pr.calculateScore(name, metrics)
        if score > bestScore {
            bestScore = score
            bestProcessor = name
        }
    }
    
    if bestProcessor == "" {
        return nil, fmt.Errorf("no healthy processors available")
    }
    
    return pr.processors[bestProcessor], nil
}

func (pr *PaymentRouter) calculateScore(processor string, metrics *RoutingMetrics) float64 {
    successRate := metrics.successRates[processor]
    avgLatency := metrics.avgLatency[processor]
    cost := metrics.costs[processor]
    
    // Weighted scoring algorithm
    score := (successRate * 0.4) + 
             ((1.0 - float64(avgLatency)/float64(time.Second)) * 0.3) +
             ((1.0 - cost) * 0.3)
    
    return score
}
```

## Financial Systems Design

### Problem 3: Design a Reconciliation System

**Problem Statement**: Design a system that reconciles transactions between Razorpay and various payment processors to ensure financial accuracy.

**Requirements**:
- Handle millions of transactions daily
- Detect discrepancies automatically
- Support multiple reconciliation types
- Provide audit trails
- Handle partial matches
- Support manual intervention

**Solution**:

```go
// Reconciliation System
type ReconciliationSystem struct {
    matcher      *TransactionMatcher
    validator    *ReconciliationValidator
    reporter     *ReconciliationReporter
    notifier     *NotificationService
    storage      *ReconciliationStorage
}

type Transaction struct {
    ID              string
    Amount          int64
    Currency        string
    Status          string
    Timestamp       time.Time
    Processor       string
    GatewayRefID    string
    MerchantID      string
    OrderID         string
    Metadata        map[string]string
}

type ReconciliationResult struct {
    ID              string
    Type            string
    Status          string
    MatchedCount    int
    UnmatchedCount  int
    Discrepancies   []*Discrepancy
    Timestamp       time.Time
}

type Discrepancy struct {
    Type            string
    Description     string
    Severity        string
    TransactionID   string
    ExpectedValue   interface{}
    ActualValue     interface{}
    Resolution      string
}

func (rs *ReconciliationSystem) Reconcile(processor string, date time.Time) (*ReconciliationResult, error) {
    // Fetch transactions from Razorpay
    razorpayTxs, err := rs.fetchRazorpayTransactions(date)
    if err != nil {
        return nil, err
    }
    
    // Fetch transactions from processor
    processorTxs, err := rs.fetchProcessorTransactions(processor, date)
    if err != nil {
        return nil, err
    }
    
    // Match transactions
    matches, unmatched := rs.matcher.Match(razorpayTxs, processorTxs)
    
    // Validate matches
    discrepancies := rs.validator.Validate(matches)
    
    // Create reconciliation result
    result := &ReconciliationResult{
        ID:             generateReconciliationID(),
        Type:           "daily",
        Status:         "completed",
        MatchedCount:   len(matches),
        UnmatchedCount: len(unmatched),
        Discrepancies:  discrepancies,
        Timestamp:      time.Now(),
    }
    
    // Store result
    if err := rs.storage.Store(result); err != nil {
        return nil, err
    }
    
    // Send notifications for discrepancies
    if len(discrepancies) > 0 {
        rs.notifier.NotifyDiscrepancies(discrepancies)
    }
    
    return result, nil
}
```

### Problem 4: Design a Fraud Detection System

**Problem Statement**: Design a real-time fraud detection system that can identify suspicious transactions within 100ms.

**Requirements**:
- Real-time processing
- Machine learning models
- Rule-based detection
- Risk scoring
- False positive minimization
- Scalable architecture

**Solution**:

```go
// Fraud Detection System
type FraudDetectionSystem struct {
    models        map[string]*MLModel
    rules         []*FraudRule
    riskEngine    *RiskEngine
    featureStore  *FeatureStore
    streamProcessor *StreamProcessor
}

type FraudRule struct {
    ID          string
    Name        string
    Condition   string
    RiskScore   float64
    Enabled     bool
    Priority    int
}

type RiskScore struct {
    TransactionID string
    Score         float64
    Factors       []*RiskFactor
    Decision      string
    Timestamp     time.Time
}

type RiskFactor struct {
    Name        string
    Value       float64
    Weight      float64
    Description string
}

func (fds *FraudDetectionSystem) AnalyzeTransaction(tx *Transaction) (*RiskScore, error) {
    // Extract features
    features := fds.featureStore.ExtractFeatures(tx)
    
    // Apply ML models
    mlScore := fds.applyMLModels(features)
    
    // Apply rules
    ruleScore := fds.applyRules(tx, features)
    
    // Calculate combined risk score
    combinedScore := fds.calculateCombinedScore(mlScore, ruleScore)
    
    // Make decision
    decision := fds.makeDecision(combinedScore)
    
    riskScore := &RiskScore{
        TransactionID: tx.ID,
        Score:         combinedScore,
        Factors:       fds.getRiskFactors(features),
        Decision:      decision,
        Timestamp:     time.Now(),
    }
    
    return riskScore, nil
}

func (fds *FraudDetectionSystem) applyMLModels(features map[string]float64) float64 {
    totalScore := 0.0
    modelCount := 0
    
    for _, model := range fds.models {
        if model.IsEnabled() {
            score := model.Predict(features)
            totalScore += score
            modelCount++
        }
    }
    
    if modelCount == 0 {
        return 0.0
    }
    
    return totalScore / float64(modelCount)
}

func (fds *FraudDetectionSystem) applyRules(tx *Transaction, features map[string]float64) float64 {
    totalScore := 0.0
    
    for _, rule := range fds.rules {
        if rule.Enabled && fds.evaluateRule(rule, tx, features) {
            totalScore += rule.RiskScore
        }
    }
    
    return totalScore
}
```

## Compliance and Security

### Problem 5: Design a PCI DSS Compliant System

**Problem Statement**: Design a system that meets PCI DSS requirements for handling cardholder data.

**Requirements**:
- Secure data storage
- Encryption in transit and at rest
- Access controls
- Audit logging
- Vulnerability management
- Network security

**Solution**:

```go
// PCI DSS Compliance System
type PCIDSSSystem struct {
    encryptor     *DataEncryptor
    tokenizer     *Tokenizer
    accessControl *AccessControl
    auditLogger   *AuditLogger
    vault         *SecureVault
    networkSec    *NetworkSecurity
}

type CardholderData struct {
    PAN           string
    ExpiryDate    string
    CVV           string
    CardholderName string
    Token         string
}

type Token struct {
    Value       string
    PAN         string
    ExpiryDate  string
    CreatedAt   time.Time
    ExpiresAt   time.Time
    UsageCount  int
}

func (pds *PCIDSSSystem) ProcessCardData(data *CardholderData) (*Token, error) {
    // Validate input
    if err := pds.validateCardData(data); err != nil {
        return nil, err
    }
    
    // Log access attempt
    pds.auditLogger.LogAccess("card_data_processing", data.PAN[:6]+"****")
    
    // Encrypt sensitive data
    encryptedData, err := pds.encryptor.Encrypt(data)
    if err != nil {
        return nil, err
    }
    
    // Store in secure vault
    if err := pds.vault.Store(encryptedData); err != nil {
        return nil, err
    }
    
    // Generate token
    token := pds.tokenizer.GenerateToken(data)
    
    // Store token mapping
    if err := pds.vault.StoreTokenMapping(token, data.PAN); err != nil {
        return nil, err
    }
    
    return token, nil
}

func (pds *PCIDSSSystem) validateCardData(data *CardholderData) error {
    // Validate PAN format
    if !pds.isValidPAN(data.PAN) {
        return fmt.Errorf("invalid PAN format")
    }
    
    // Validate expiry date
    if !pds.isValidExpiryDate(data.ExpiryDate) {
        return fmt.Errorf("invalid expiry date")
    }
    
    // Validate CVV
    if !pds.isValidCVV(data.CVV) {
        return fmt.Errorf("invalid CVV")
    }
    
    return nil
}
```

## Scalability Challenges

### Problem 6: Design a High-Throughput Payment System

**Problem Statement**: Design a payment system that can handle 100,000 transactions per second during peak hours.

**Requirements**:
- Horizontal scaling
- Load balancing
- Database optimization
- Caching strategies
- Message queuing
- Circuit breakers

**Solution**:

```go
// High-Throughput Payment System
type HighThroughputPaymentSystem struct {
    loadBalancer  *LoadBalancer
    paymentNodes  []*PaymentNode
    messageQueue  *MessageQueue
    cache         *DistributedCache
    database      *ShardedDatabase
    circuitBreaker *CircuitBreaker
}

type PaymentNode struct {
    ID            string
    Capacity      int
    CurrentLoad   int
    Health        string
    Processors    []*PaymentProcessor
    mu            sync.RWMutex
}

func (htps *HighThroughputPaymentSystem) ProcessPayment(request *PaymentRequest) (*PaymentResponse, error) {
    // Check circuit breaker
    if !htps.circuitBreaker.Allow() {
        return nil, fmt.Errorf("circuit breaker open")
    }
    
    // Select payment node
    node := htps.loadBalancer.SelectNode()
    if node == nil {
        return nil, fmt.Errorf("no available nodes")
    }
    
    // Check cache first
    if cached, exists := htps.cache.Get(request.ID); exists {
        return cached.(*PaymentResponse), nil
    }
    
    // Process payment
    response, err := node.ProcessPayment(request)
    if err != nil {
        htps.circuitBreaker.RecordFailure()
        return nil, err
    }
    
    // Cache successful response
    htps.cache.Set(request.ID, response, time.Minute*5)
    
    // Record success
    htps.circuitBreaker.RecordSuccess()
    
    return response, nil
}

func (htps *HighThroughputPaymentSystem) ScaleOut() error {
    // Create new payment node
    newNode := htps.createPaymentNode()
    
    // Add to load balancer
    htps.loadBalancer.AddNode(newNode)
    
    // Update monitoring
    htps.updateMonitoring()
    
    return nil
}
```

## Real-Time Systems

### Problem 7: Design a Real-Time Settlement System

**Problem Statement**: Design a system that processes settlements in real-time with sub-second latency.

**Requirements**:
- Real-time processing
- Event-driven architecture
- Stream processing
- Exactly-once semantics
- Monitoring and alerting

**Solution**:

```go
// Real-Time Settlement System
type RealTimeSettlementSystem struct {
    streamProcessor *StreamProcessor
    eventStore      *EventStore
    settlementEngine *SettlementEngine
    notificationService *NotificationService
    monitoring      *Monitoring
}

type SettlementEvent struct {
    ID              string
    Type            string
    TransactionID   string
    Amount          int64
    Currency        string
    MerchantID      string
    ProcessorID     string
    Timestamp       time.Time
    Status          string
}

func (rtss *RealTimeSettlementSystem) ProcessSettlement(event *SettlementEvent) error {
    // Validate event
    if err := rtss.validateEvent(event); err != nil {
        return err
    }
    
    // Process settlement
    settlement, err := rtss.settlementEngine.Process(event)
    if err != nil {
        return err
    }
    
    // Store event
    if err := rtss.eventStore.Store(event); err != nil {
        return err
    }
    
    // Send notifications
    if err := rtss.notificationService.NotifySettlement(settlement); err != nil {
        log.Printf("Failed to send notification: %v", err)
    }
    
    // Update monitoring
    rtss.monitoring.RecordSettlement(settlement)
    
    return nil
}

func (rtss *RealTimeSettlementSystem) validateEvent(event *SettlementEvent) error {
    if event.ID == "" {
        return fmt.Errorf("event ID is required")
    }
    
    if event.Amount <= 0 {
        return fmt.Errorf("invalid amount")
    }
    
    if event.TransactionID == "" {
        return fmt.Errorf("transaction ID is required")
    }
    
    return nil
}
```

## Data Engineering

### Problem 8: Design a Financial Data Pipeline

**Problem Statement**: Design a data pipeline that processes financial transactions and generates real-time analytics.

**Requirements**:
- Handle high-volume data
- Real-time processing
- Data quality checks
- Schema evolution
- Monitoring and alerting

**Solution**:

```go
// Financial Data Pipeline
type FinancialDataPipeline struct {
    ingestors     []*DataIngestor
    processors    []*DataProcessor
    validators    []*DataValidator
    transformers  []*DataTransformer
    sinks         []*DataSink
    monitoring    *PipelineMonitoring
}

type DataIngestor struct {
    Name        string
    Source      string
    Format      string
    BatchSize   int
    RateLimit   int
    Health      string
}

type DataProcessor struct {
    Name        string
    Function    func(interface{}) interface{}
    Parallelism int
    Buffer      *DataBuffer
}

func (fdp *FinancialDataPipeline) ProcessData(data []byte) error {
    // Ingest data
    records, err := fdp.ingestData(data)
    if err != nil {
        return err
    }
    
    // Validate data
    validRecords, err := fdp.validateData(records)
    if err != nil {
        return err
    }
    
    // Transform data
    transformedRecords, err := fdp.transformData(validRecords)
    if err != nil {
        return err
    }
    
    // Process data
    processedRecords, err := fdp.processData(transformedRecords)
    if err != nil {
        return err
    }
    
    // Sink data
    if err := fdp.sinkData(processedRecords); err != nil {
        return err
    }
    
    return nil
}

func (fdp *FinancialDataPipeline) ingestData(data []byte) ([]*Record, error) {
    records := make([]*Record, 0)
    
    for _, ingestor := range fdp.ingestors {
        if ingestor.Health == "healthy" {
            record, err := ingestor.Ingest(data)
            if err != nil {
                continue
            }
            records = append(records, record)
        }
    }
    
    return records, nil
}
```

## Machine Learning Applications

### Problem 9: Design a ML-Powered Risk Assessment System

**Problem Statement**: Design a machine learning system that assesses risk for financial transactions in real-time.

**Requirements**:
- Real-time inference
- Model versioning
- A/B testing
- Feature engineering
- Model monitoring

**Solution**:

```go
// ML Risk Assessment System
type MLRiskAssessmentSystem struct {
    models        map[string]*MLModel
    featureStore  *FeatureStore
    modelRegistry *ModelRegistry
    abTestManager *ABTestManager
    monitoring    *ModelMonitoring
}

type MLModel struct {
    ID          string
    Version     string
    Type        string
    Path        string
    Features    []string
    Performance *ModelPerformance
    Status      string
}

type ModelPerformance struct {
    Accuracy    float64
    Precision   float64
    Recall      float64
    F1Score     float64
    AUC         float64
    LastUpdated time.Time
}

func (mlras *MLRiskAssessmentSystem) AssessRisk(transaction *Transaction) (*RiskAssessment, error) {
    // Extract features
    features := mlras.featureStore.ExtractFeatures(transaction)
    
    // Select model for A/B testing
    model := mlras.abTestManager.SelectModel(transaction)
    
    // Make prediction
    prediction, err := model.Predict(features)
    if err != nil {
        return nil, err
    }
    
    // Create risk assessment
    assessment := &RiskAssessment{
        TransactionID: transaction.ID,
        RiskScore:     prediction.Score,
        RiskLevel:     mlras.calculateRiskLevel(prediction.Score),
        ModelID:       model.ID,
        ModelVersion:  model.Version,
        Features:      features,
        Timestamp:     time.Now(),
    }
    
    // Monitor prediction
    mlras.monitoring.RecordPrediction(assessment)
    
    return assessment, nil
}

func (mlras *MLRiskAssessmentSystem) calculateRiskLevel(score float64) string {
    if score < 0.3 {
        return "low"
    } else if score < 0.7 {
        return "medium"
    } else {
        return "high"
    }
}
```

## Conclusion

Razorpay-specific interview problems test your understanding of:

1. **Payment Processing**: Gateway design, routing, and processing
2. **Financial Systems**: Reconciliation, fraud detection, and compliance
3. **Scalability**: High-throughput systems and load balancing
4. **Real-Time Systems**: Event-driven architecture and stream processing
5. **Data Engineering**: Pipelines and analytics
6. **Machine Learning**: Risk assessment and fraud detection
7. **Security**: PCI DSS compliance and data protection
8. **Monitoring**: Observability and alerting

These problems require a deep understanding of both technical and business aspects of fintech systems, making them excellent preparation for senior engineering roles at financial technology companies.

## Additional Resources

- [Razorpay Engineering Blog](https://engineering.razorpay.com/)
- [Payment Gateway Integration](https://razorpay.com/docs/)
- [PCI DSS Requirements](https://www.pcisecuritystandards.org/)
- [Financial Technology Trends](https://www.fintechnews.org/)
- [Payment Processing Best Practices](https://www.paymentprocessing.com/)
- [Fraud Detection in Fintech](https://www.frauddetection.com/)
- [Real-Time Payment Systems](https://www.realtimepayments.com/)
- [Financial Data Analytics](https://www.financialanalytics.com/)
