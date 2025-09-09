# Mediator Pattern

## Pattern Name & Intent

**Mediator** is a behavioral design pattern that defines how a set of objects interact with each other. Instead of objects communicating directly, they communicate through a central mediator object. This promotes loose coupling by keeping objects from referring to each other explicitly.

**Key Intent:**
- Define how a set of objects interact without tight coupling
- Centralize complex communications and control logic
- Promote loose coupling between communicating objects
- Make object interaction easier to understand and maintain
- Enable reusable components that don't depend on each other
- Facilitate many-to-many relationships through one-to-many relationships

## When to Use

**Use Mediator when:**

1. **Complex Interactions**: Objects interact in complex ways with multiple dependencies
2. **Tight Coupling**: Objects are tightly coupled and hard to reuse independently
3. **Communication Logic**: Communication logic between objects becomes complex
4. **Many-to-Many Relationships**: Need to manage many-to-many object relationships
5. **Centralized Control**: Need centralized control over object interactions
6. **Workflow Orchestration**: Coordinating complex workflows or business processes
7. **Event Coordination**: Managing complex event-driven interactions

**Don't use when:**
- Simple one-to-one or one-to-many relationships
- Objects naturally belong together in a hierarchy
- The mediator itself becomes too complex (God object)
- Performance is critical and direct communication is faster
- The system is very simple with few interactions

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Processing Orchestrator
```go
// Payment processing involves multiple services that need coordination
type PaymentMediator interface {
    ProcessPayment(ctx context.Context, request *PaymentRequest) (*PaymentResult, error)
    HandlePaymentUpdate(ctx context.Context, update *PaymentUpdate) error
    RegisterComponent(component PaymentComponent) error
    UnregisterComponent(componentID string) error
}

// Components that participate in payment processing
type PaymentComponent interface {
    GetComponentID() string
    GetComponentType() string
    HandleNotification(ctx context.Context, notification *PaymentNotification) error
}

// Central mediator for payment processing
type PaymentProcessingMediator struct {
    components     map[string]PaymentComponent
    validator      ValidationService
    fraudDetector  FraudDetectionService
    gateway        PaymentGateway
    notifications  NotificationService
    audit         AuditService
    logger        *zap.Logger
    mu            sync.RWMutex
}

func NewPaymentProcessingMediator(logger *zap.Logger) *PaymentProcessingMediator {
    return &PaymentProcessingMediator{
        components: make(map[string]PaymentComponent),
        logger:     logger,
    }
}

func (ppm *PaymentProcessingMediator) RegisterComponent(component PaymentComponent) error {
    ppm.mu.Lock()
    defer ppm.mu.Unlock()
    
    componentID := component.GetComponentID()
    if _, exists := ppm.components[componentID]; exists {
        return fmt.Errorf("component %s already registered", componentID)
    }
    
    ppm.components[componentID] = component
    
    // Set up specific components
    switch component.GetComponentType() {
    case "VALIDATOR":
        ppm.validator = component.(ValidationService)
    case "FRAUD_DETECTOR":
        ppm.fraudDetector = component.(FraudDetectionService)
    case "GATEWAY":
        ppm.gateway = component.(PaymentGateway)
    case "NOTIFICATIONS":
        ppm.notifications = component.(NotificationService)
    case "AUDIT":
        ppm.audit = component.(AuditService)
    }
    
    ppm.logger.Info("Component registered", 
        zap.String("component_id", componentID),
        zap.String("component_type", component.GetComponentType()))
    
    return nil
}

func (ppm *PaymentProcessingMediator) ProcessPayment(ctx context.Context, request *PaymentRequest) (*PaymentResult, error) {
    ppm.logger.Info("Starting payment processing", 
        zap.String("payment_id", request.PaymentID))
    
    // Step 1: Validation
    if err := ppm.validatePayment(ctx, request); err != nil {
        ppm.notifyComponents(ctx, &PaymentNotification{
            Type:      "VALIDATION_FAILED",
            PaymentID: request.PaymentID,
            Error:     err.Error(),
        })
        return nil, fmt.Errorf("validation failed: %w", err)
    }
    
    // Step 2: Fraud Detection
    riskScore, err := ppm.performFraudDetection(ctx, request)
    if err != nil {
        ppm.notifyComponents(ctx, &PaymentNotification{
            Type:      "FRAUD_CHECK_FAILED",
            PaymentID: request.PaymentID,
            Error:     err.Error(),
        })
        return nil, fmt.Errorf("fraud detection failed: %w", err)
    }
    
    if riskScore > 0.8 {
        ppm.notifyComponents(ctx, &PaymentNotification{
            Type:      "HIGH_RISK_DETECTED",
            PaymentID: request.PaymentID,
            Data:      map[string]interface{}{"risk_score": riskScore},
        })
        return nil, fmt.Errorf("payment blocked due to high risk")
    }
    
    // Step 3: Process Payment
    result, err := ppm.processWithGateway(ctx, request)
    if err != nil {
        ppm.notifyComponents(ctx, &PaymentNotification{
            Type:      "PAYMENT_FAILED",
            PaymentID: request.PaymentID,
            Error:     err.Error(),
        })
        return nil, fmt.Errorf("payment processing failed: %w", err)
    }
    
    // Step 4: Success Notifications
    ppm.notifyComponents(ctx, &PaymentNotification{
        Type:      "PAYMENT_SUCCESSFUL",
        PaymentID: request.PaymentID,
        Data: map[string]interface{}{
            "transaction_id": result.TransactionID,
            "amount":         result.Amount,
            "risk_score":     riskScore,
        },
    })
    
    ppm.logger.Info("Payment processing completed", 
        zap.String("payment_id", request.PaymentID),
        zap.String("transaction_id", result.TransactionID))
    
    return result, nil
}

func (ppm *PaymentProcessingMediator) validatePayment(ctx context.Context, request *PaymentRequest) error {
    if ppm.validator == nil {
        return fmt.Errorf("validation service not available")
    }
    
    return ppm.validator.ValidatePayment(ctx, request)
}

func (ppm *PaymentProcessingMediator) performFraudDetection(ctx context.Context, request *PaymentRequest) (float64, error) {
    if ppm.fraudDetector == nil {
        return 0.0, fmt.Errorf("fraud detection service not available")
    }
    
    return ppm.fraudDetector.CalculateRiskScore(ctx, request)
}

func (ppm *PaymentProcessingMediator) processWithGateway(ctx context.Context, request *PaymentRequest) (*PaymentResult, error) {
    if ppm.gateway == nil {
        return nil, fmt.Errorf("payment gateway not available")
    }
    
    return ppm.gateway.ProcessPayment(ctx, request)
}

func (ppm *PaymentProcessingMediator) notifyComponents(ctx context.Context, notification *PaymentNotification) {
    ppm.mu.RLock()
    defer ppm.mu.RUnlock()
    
    for _, component := range ppm.components {
        go func(comp PaymentComponent) {
            if err := comp.HandleNotification(ctx, notification); err != nil {
                ppm.logger.Warn("Component notification failed", 
                    zap.String("component_id", comp.GetComponentID()),
                    zap.Error(err))
            }
        }(component)
    }
}

func (ppm *PaymentProcessingMediator) HandlePaymentUpdate(ctx context.Context, update *PaymentUpdate) error {
    ppm.logger.Info("Handling payment update", 
        zap.String("payment_id", update.PaymentID),
        zap.String("status", update.Status))
    
    notification := &PaymentNotification{
        Type:      "PAYMENT_UPDATE",
        PaymentID: update.PaymentID,
        Data: map[string]interface{}{
            "status":     update.Status,
            "updated_at": update.UpdatedAt,
        },
    }
    
    ppm.notifyComponents(ctx, notification)
    
    // Handle specific status updates
    switch update.Status {
    case "REFUNDED":
        return ppm.handleRefund(ctx, update)
    case "CHARGEBACK":
        return ppm.handleChargeback(ctx, update)
    case "SETTLED":
        return ppm.handleSettlement(ctx, update)
    }
    
    return nil
}

func (ppm *PaymentProcessingMediator) handleRefund(ctx context.Context, update *PaymentUpdate) error {
    ppm.logger.Info("Processing refund", zap.String("payment_id", update.PaymentID))
    
    // Notify all relevant components about refund
    notification := &PaymentNotification{
        Type:      "REFUND_PROCESSED",
        PaymentID: update.PaymentID,
        Data:      map[string]interface{}{"refund_amount": update.Amount},
    }
    
    ppm.notifyComponents(ctx, notification)
    return nil
}

func (ppm *PaymentProcessingMediator) handleChargeback(ctx context.Context, update *PaymentUpdate) error {
    ppm.logger.Warn("Processing chargeback", zap.String("payment_id", update.PaymentID))
    
    // Notify components about chargeback for appropriate handling
    notification := &PaymentNotification{
        Type:      "CHARGEBACK_RECEIVED",
        PaymentID: update.PaymentID,
        Data:      map[string]interface{}{"chargeback_amount": update.Amount},
    }
    
    ppm.notifyComponents(ctx, notification)
    return nil
}

func (ppm *PaymentProcessingMediator) handleSettlement(ctx context.Context, update *PaymentUpdate) error {
    ppm.logger.Info("Processing settlement", zap.String("payment_id", update.PaymentID))
    
    notification := &PaymentNotification{
        Type:      "PAYMENT_SETTLED",
        PaymentID: update.PaymentID,
        Data:      map[string]interface{}{"settled_amount": update.Amount},
    }
    
    ppm.notifyComponents(ctx, notification)
    return nil
}

// Concrete components
type ValidationComponent struct {
    componentID string
    rules       []ValidationRule
    logger      *zap.Logger
}

func NewValidationComponent(componentID string, logger *zap.Logger) *ValidationComponent {
    return &ValidationComponent{
        componentID: componentID,
        rules:       make([]ValidationRule, 0),
        logger:      logger,
    }
}

func (vc *ValidationComponent) GetComponentID() string {
    return vc.componentID
}

func (vc *ValidationComponent) GetComponentType() string {
    return "VALIDATOR"
}

func (vc *ValidationComponent) ValidatePayment(ctx context.Context, request *PaymentRequest) error {
    for _, rule := range vc.rules {
        if err := rule.Validate(request); err != nil {
            return err
        }
    }
    return nil
}

func (vc *ValidationComponent) HandleNotification(ctx context.Context, notification *PaymentNotification) error {
    switch notification.Type {
    case "PAYMENT_SUCCESSFUL":
        vc.logger.Info("Payment validation succeeded", 
            zap.String("payment_id", notification.PaymentID))
    case "VALIDATION_FAILED":
        vc.logger.Warn("Payment validation failed", 
            zap.String("payment_id", notification.PaymentID))
    }
    return nil
}

// Fraud Detection Component
type FraudDetectionComponent struct {
    componentID string
    engine      FraudEngine
    logger      *zap.Logger
}

func NewFraudDetectionComponent(componentID string, engine FraudEngine, logger *zap.Logger) *FraudDetectionComponent {
    return &FraudDetectionComponent{
        componentID: componentID,
        engine:      engine,
        logger:      logger,
    }
}

func (fdc *FraudDetectionComponent) GetComponentID() string {
    return fdc.componentID
}

func (fdc *FraudDetectionComponent) GetComponentType() string {
    return "FRAUD_DETECTOR"
}

func (fdc *FraudDetectionComponent) CalculateRiskScore(ctx context.Context, request *PaymentRequest) (float64, error) {
    return fdc.engine.CalculateRisk(ctx, request)
}

func (fdc *FraudDetectionComponent) HandleNotification(ctx context.Context, notification *PaymentNotification) error {
    switch notification.Type {
    case "HIGH_RISK_DETECTED":
        fdc.logger.Warn("High risk transaction detected", 
            zap.String("payment_id", notification.PaymentID))
        // Update fraud models, trigger additional checks, etc.
    case "CHARGEBACK_RECEIVED":
        fdc.logger.Warn("Chargeback received, updating fraud models", 
            zap.String("payment_id", notification.PaymentID))
        // Use chargeback data to improve fraud detection
    }
    return nil
}

// Audit Component
type AuditComponent struct {
    componentID string
    storage     AuditStorage
    logger      *zap.Logger
}

func NewAuditComponent(componentID string, storage AuditStorage, logger *zap.Logger) *AuditComponent {
    return &AuditComponent{
        componentID: componentID,
        storage:     storage,
        logger:      logger,
    }
}

func (ac *AuditComponent) GetComponentID() string {
    return ac.componentID
}

func (ac *AuditComponent) GetComponentType() string {
    return "AUDIT"
}

func (ac *AuditComponent) HandleNotification(ctx context.Context, notification *PaymentNotification) error {
    // Log all payment events for audit trail
    auditEntry := &AuditEntry{
        EventType:   notification.Type,
        PaymentID:   notification.PaymentID,
        Timestamp:   time.Now(),
        Data:        notification.Data,
        Error:       notification.Error,
    }
    
    if err := ac.storage.StoreAuditEntry(ctx, auditEntry); err != nil {
        ac.logger.Error("Failed to store audit entry", zap.Error(err))
        return err
    }
    
    ac.logger.Debug("Audit entry stored", 
        zap.String("event_type", notification.Type),
        zap.String("payment_id", notification.PaymentID))
    
    return nil
}

// Supporting types
type PaymentRequest struct {
    PaymentID     string
    Amount        decimal.Decimal
    Currency      string
    PaymentMethod string
    CustomerID    string
    MerchantID    string
    Description   string
    Metadata      map[string]interface{}
}

type PaymentResult struct {
    PaymentID     string
    TransactionID string
    Amount        decimal.Decimal
    Status        string
    ProcessedAt   time.Time
}

type PaymentUpdate struct {
    PaymentID string
    Status    string
    Amount    decimal.Decimal
    UpdatedAt time.Time
    Metadata  map[string]interface{}
}

type PaymentNotification struct {
    Type      string
    PaymentID string
    Data      map[string]interface{}
    Error     string
    Timestamp time.Time
}

type AuditEntry struct {
    EventType string
    PaymentID string
    Timestamp time.Time
    Data      map[string]interface{}
    Error     string
}
```

### 2. Trading System Mediator
```go
// Trading system with multiple participants and complex interactions
type TradingMediator interface {
    PlaceOrder(ctx context.Context, order *Order) (*OrderResult, error)
    CancelOrder(ctx context.Context, orderID string) error
    MatchOrders(ctx context.Context) error
    HandleMarketData(ctx context.Context, data *MarketData) error
    RegisterParticipant(participant TradingParticipant) error
}

type TradingParticipant interface {
    GetParticipantID() string
    GetParticipantType() string
    HandleTradingEvent(ctx context.Context, event *TradingEvent) error
}

// Central trading mediator
type CentralTradingMediator struct {
    participants    map[string]TradingParticipant
    orderBook      OrderBook
    matchingEngine MatchingEngine
    riskManager    RiskManager
    marketData     MarketDataProvider
    settlement     SettlementService
    logger         *zap.Logger
    mu             sync.RWMutex
}

func NewCentralTradingMediator(logger *zap.Logger) *CentralTradingMediator {
    return &CentralTradingMediator{
        participants: make(map[string]TradingParticipant),
        logger:       logger,
    }
}

func (ctm *CentralTradingMediator) PlaceOrder(ctx context.Context, order *Order) (*OrderResult, error) {
    ctm.logger.Info("Order placement started", 
        zap.String("order_id", order.OrderID),
        zap.String("symbol", order.Symbol),
        zap.String("side", order.Side))
    
    // Step 1: Risk Check
    if err := ctm.riskManager.CheckOrderRisk(ctx, order); err != nil {
        ctm.notifyParticipants(ctx, &TradingEvent{
            Type:    "ORDER_REJECTED",
            OrderID: order.OrderID,
            Reason:  "Risk check failed: " + err.Error(),
        })
        return nil, fmt.Errorf("risk check failed: %w", err)
    }
    
    // Step 2: Add to Order Book
    if err := ctm.orderBook.AddOrder(order); err != nil {
        ctm.notifyParticipants(ctx, &TradingEvent{
            Type:    "ORDER_REJECTED",
            OrderID: order.OrderID,
            Reason:  "Order book error: " + err.Error(),
        })
        return nil, fmt.Errorf("order book error: %w", err)
    }
    
    // Step 3: Notify Order Acceptance
    ctm.notifyParticipants(ctx, &TradingEvent{
        Type:    "ORDER_ACCEPTED",
        OrderID: order.OrderID,
        Data: map[string]interface{}{
            "symbol":    order.Symbol,
            "quantity":  order.Quantity,
            "price":     order.Price,
            "side":      order.Side,
        },
    })
    
    // Step 4: Attempt Matching
    matches, err := ctm.matchingEngine.FindMatches(ctx, order)
    if err != nil {
        ctm.logger.Error("Matching engine error", zap.Error(err))
        return &OrderResult{OrderID: order.OrderID, Status: "PENDING"}, nil
    }
    
    // Step 5: Process Matches
    for _, match := range matches {
        if err := ctm.processMatch(ctx, match); err != nil {
            ctm.logger.Error("Match processing failed", zap.Error(err))
        }
    }
    
    return &OrderResult{
        OrderID:   order.OrderID,
        Status:    "ACCEPTED",
        Matches:   matches,
        Timestamp: time.Now(),
    }, nil
}

func (ctm *CentralTradingMediator) processMatch(ctx context.Context, match *OrderMatch) error {
    ctm.logger.Info("Processing order match", 
        zap.String("buy_order", match.BuyOrder.OrderID),
        zap.String("sell_order", match.SellOrder.OrderID),
        zap.String("quantity", match.Quantity.String()))
    
    // Create trade
    trade := &Trade{
        TradeID:       generateTradeID(),
        BuyOrderID:    match.BuyOrder.OrderID,
        SellOrderID:   match.SellOrder.OrderID,
        Symbol:        match.BuyOrder.Symbol,
        Quantity:      match.Quantity,
        Price:         match.Price,
        Timestamp:     time.Now(),
    }
    
    // Notify trade execution
    ctm.notifyParticipants(ctx, &TradingEvent{
        Type:    "TRADE_EXECUTED",
        TradeID: trade.TradeID,
        Data: map[string]interface{}{
            "symbol":       trade.Symbol,
            "quantity":     trade.Quantity,
            "price":        trade.Price,
            "buy_order":    trade.BuyOrderID,
            "sell_order":   trade.SellOrderID,
        },
    })
    
    // Initiate settlement
    if err := ctm.settlement.SettleTrade(ctx, trade); err != nil {
        ctm.logger.Error("Settlement initiation failed", zap.Error(err))
        return err
    }
    
    // Update market data
    if err := ctm.marketData.UpdateLastTrade(ctx, trade); err != nil {
        ctm.logger.Warn("Market data update failed", zap.Error(err))
    }
    
    return nil
}

func (ctm *CentralTradingMediator) HandleMarketData(ctx context.Context, data *MarketData) error {
    ctm.logger.Debug("Market data received", 
        zap.String("symbol", data.Symbol),
        zap.String("price", data.LastPrice.String()))
    
    // Notify all participants about market data update
    ctm.notifyParticipants(ctx, &TradingEvent{
        Type:   "MARKET_DATA_UPDATE",
        Symbol: data.Symbol,
        Data: map[string]interface{}{
            "last_price": data.LastPrice,
            "volume":     data.Volume,
            "timestamp":  data.Timestamp,
        },
    })
    
    // Check if any stop orders should be triggered
    if err := ctm.checkStopOrders(ctx, data); err != nil {
        ctm.logger.Error("Stop order check failed", zap.Error(err))
    }
    
    return nil
}

func (ctm *CentralTradingMediator) checkStopOrders(ctx context.Context, data *MarketData) error {
    stopOrders := ctm.orderBook.GetStopOrders(data.Symbol)
    
    for _, order := range stopOrders {
        triggered := false
        
        if order.Side == "BUY" && data.LastPrice.GreaterThanOrEqual(order.StopPrice) {
            triggered = true
        } else if order.Side == "SELL" && data.LastPrice.LessThanOrEqual(order.StopPrice) {
            triggered = true
        }
        
        if triggered {
            ctm.logger.Info("Stop order triggered", 
                zap.String("order_id", order.OrderID),
                zap.String("stop_price", order.StopPrice.String()),
                zap.String("market_price", data.LastPrice.String()))
            
            // Convert stop order to market order
            marketOrder := order.ConvertToMarketOrder()
            if _, err := ctm.PlaceOrder(ctx, marketOrder); err != nil {
                ctm.logger.Error("Failed to place stop order as market order", zap.Error(err))
            }
            
            // Remove from stop orders
            ctm.orderBook.RemoveStopOrder(order.OrderID)
        }
    }
    
    return nil
}

func (ctm *CentralTradingMediator) notifyParticipants(ctx context.Context, event *TradingEvent) {
    ctm.mu.RLock()
    defer ctm.mu.RUnlock()
    
    for _, participant := range ctm.participants {
        go func(p TradingParticipant) {
            if err := p.HandleTradingEvent(ctx, event); err != nil {
                ctm.logger.Warn("Participant notification failed", 
                    zap.String("participant_id", p.GetParticipantID()),
                    zap.Error(err))
            }
        }(participant)
    }
}

// Trading participants
type Trader struct {
    participantID string
    portfolio     Portfolio
    riskLimits    RiskLimits
    logger        *zap.Logger
}

func NewTrader(participantID string, logger *zap.Logger) *Trader {
    return &Trader{
        participantID: participantID,
        portfolio:     NewPortfolio(),
        logger:        logger,
    }
}

func (t *Trader) GetParticipantID() string {
    return t.participantID
}

func (t *Trader) GetParticipantType() string {
    return "TRADER"
}

func (t *Trader) HandleTradingEvent(ctx context.Context, event *TradingEvent) error {
    switch event.Type {
    case "TRADE_EXECUTED":
        return t.handleTradeExecution(ctx, event)
    case "ORDER_REJECTED":
        return t.handleOrderRejection(ctx, event)
    case "MARKET_DATA_UPDATE":
        return t.handleMarketDataUpdate(ctx, event)
    }
    return nil
}

func (t *Trader) handleTradeExecution(ctx context.Context, event *TradingEvent) error {
    t.logger.Info("Trade executed notification received", 
        zap.String("trade_id", event.TradeID))
    
    // Update portfolio based on trade
    // Implementation would update positions, calculate P&L, etc.
    
    return nil
}

func (t *Trader) handleOrderRejection(ctx context.Context, event *TradingEvent) error {
    t.logger.Warn("Order rejected", 
        zap.String("order_id", event.OrderID),
        zap.String("reason", event.Reason))
    
    // Handle rejection logic - might retry, adjust strategy, etc.
    
    return nil
}

func (t *Trader) handleMarketDataUpdate(ctx context.Context, event *TradingEvent) error {
    // Update internal market view, trigger algorithmic trading decisions, etc.
    return nil
}

// Market Maker participant
type MarketMaker struct {
    participantID string
    quotingEngine QuotingEngine
    inventory     Inventory
    logger        *zap.Logger
}

func NewMarketMaker(participantID string, logger *zap.Logger) *MarketMaker {
    return &MarketMaker{
        participantID: participantID,
        logger:        logger,
    }
}

func (mm *MarketMaker) GetParticipantID() string {
    return mm.participantID
}

func (mm *MarketMaker) GetParticipantType() string {
    return "MARKET_MAKER"
}

func (mm *MarketMaker) HandleTradingEvent(ctx context.Context, event *TradingEvent) error {
    switch event.Type {
    case "MARKET_DATA_UPDATE":
        return mm.updateQuotes(ctx, event)
    case "TRADE_EXECUTED":
        return mm.handleTradeExecution(ctx, event)
    }
    return nil
}

func (mm *MarketMaker) updateQuotes(ctx context.Context, event *TradingEvent) error {
    // Market makers update their quotes based on market data
    mm.logger.Debug("Updating quotes based on market data", 
        zap.String("symbol", event.Symbol))
    
    // Implementation would calculate new bid/ask prices and place orders
    
    return nil
}

func (mm *MarketMaker) handleTradeExecution(ctx context.Context, event *TradingEvent) error {
    // Update inventory and risk position
    mm.logger.Info("Updating inventory after trade", 
        zap.String("trade_id", event.TradeID))
    
    return nil
}
```

### 3. Loan Approval Workflow Mediator
```go
// Loan approval involves multiple departments and decision points
type LoanApprovalMediator interface {
    SubmitApplication(ctx context.Context, app *LoanApplication) (*ApplicationResult, error)
    ProcessApplicationStep(ctx context.Context, stepResult *StepResult) error
    GetApplicationStatus(ctx context.Context, applicationID string) (*ApplicationStatus, error)
    RegisterProcessor(processor LoanProcessor) error
}

type LoanProcessor interface {
    GetProcessorID() string
    GetProcessorType() string
    CanProcess(application *LoanApplication) bool
    ProcessApplication(ctx context.Context, application *LoanApplication) (*StepResult, error)
    HandleNotification(ctx context.Context, notification *LoanNotification) error
}

// Central loan approval mediator
type LoanApprovalWorkflowMediator struct {
    processors    map[string]LoanProcessor
    applications  map[string]*LoanApplication
    workflows     map[string]*ApprovalWorkflow
    logger        *zap.Logger
    mu            sync.RWMutex
}

func NewLoanApprovalWorkflowMediator(logger *zap.Logger) *LoanApprovalWorkflowMediator {
    return &LoanApprovalWorkflowMediator{
        processors:   make(map[string]LoanProcessor),
        applications: make(map[string]*LoanApplication),
        workflows:    make(map[string]*ApprovalWorkflow),
        logger:       logger,
    }
}

func (lawm *LoanApprovalWorkflowMediator) SubmitApplication(ctx context.Context, app *LoanApplication) (*ApplicationResult, error) {
    lawm.logger.Info("Loan application submitted", 
        zap.String("application_id", app.ApplicationID),
        zap.String("applicant", app.ApplicantName),
        zap.String("amount", app.RequestedAmount.String()))
    
    // Store application
    lawm.mu.Lock()
    lawm.applications[app.ApplicationID] = app
    lawm.mu.Unlock()
    
    // Create workflow
    workflow := &ApprovalWorkflow{
        ApplicationID: app.ApplicationID,
        CurrentStep:   "INITIAL_REVIEW",
        Status:        "IN_PROGRESS",
        Steps:         lawm.defineWorkflowSteps(app),
        StartTime:     time.Now(),
    }
    
    lawm.mu.Lock()
    lawm.workflows[app.ApplicationID] = workflow
    lawm.mu.Unlock()
    
    // Notify all processors about new application
    lawm.notifyProcessors(ctx, &LoanNotification{
        Type:          "APPLICATION_SUBMITTED",
        ApplicationID: app.ApplicationID,
        Data: map[string]interface{}{
            "amount":         app.RequestedAmount,
            "applicant_name": app.ApplicantName,
            "loan_type":      app.LoanType,
        },
    })
    
    // Start processing with first applicable processor
    if err := lawm.processNextStep(ctx, app.ApplicationID); err != nil {
        return nil, fmt.Errorf("failed to start processing: %w", err)
    }
    
    return &ApplicationResult{
        ApplicationID: app.ApplicationID,
        Status:        "SUBMITTED",
        WorkflowID:    workflow.ApplicationID,
        SubmittedAt:   time.Now(),
    }, nil
}

func (lawm *LoanApprovalWorkflowMediator) processNextStep(ctx context.Context, applicationID string) error {
    lawm.mu.RLock()
    workflow, exists := lawm.workflows[applicationID]
    if !exists {
        lawm.mu.RUnlock()
        return fmt.Errorf("workflow not found for application %s", applicationID)
    }
    
    application, exists := lawm.applications[applicationID]
    if !exists {
        lawm.mu.RUnlock()
        return fmt.Errorf("application not found: %s", applicationID)
    }
    lawm.mu.RUnlock()
    
    currentStep := workflow.GetCurrentStep()
    if currentStep == nil {
        lawm.logger.Info("Workflow completed", zap.String("application_id", applicationID))
        return lawm.completeWorkflow(ctx, applicationID)
    }
    
    // Find processor for current step
    processor := lawm.findProcessorForStep(currentStep.StepType, application)
    if processor == nil {
        return fmt.Errorf("no processor found for step %s", currentStep.StepType)
    }
    
    lawm.logger.Info("Processing workflow step", 
        zap.String("application_id", applicationID),
        zap.String("step", currentStep.StepType),
        zap.String("processor", processor.GetProcessorID()))
    
    // Process asynchronously
    go func() {
        stepResult, err := processor.ProcessApplication(ctx, application)
        if err != nil {
            lawm.logger.Error("Step processing failed", 
                zap.String("application_id", applicationID),
                zap.String("step", currentStep.StepType),
                zap.Error(err))
            
            stepResult = &StepResult{
                ApplicationID: applicationID,
                StepType:      currentStep.StepType,
                Status:        "FAILED",
                Error:         err.Error(),
                ProcessedAt:   time.Now(),
            }
        }
        
        lawm.ProcessApplicationStep(ctx, stepResult)
    }()
    
    return nil
}

func (lawm *LoanApprovalWorkflowMediator) ProcessApplicationStep(ctx context.Context, stepResult *StepResult) error {
    lawm.logger.Info("Processing step result", 
        zap.String("application_id", stepResult.ApplicationID),
        zap.String("step", stepResult.StepType),
        zap.String("status", stepResult.Status))
    
    lawm.mu.Lock()
    workflow := lawm.workflows[stepResult.ApplicationID]
    lawm.mu.Unlock()
    
    if workflow == nil {
        return fmt.Errorf("workflow not found")
    }
    
    // Update workflow with step result
    workflow.UpdateStep(stepResult)
    
    // Notify processors about step completion
    lawm.notifyProcessors(ctx, &LoanNotification{
        Type:          "STEP_COMPLETED",
        ApplicationID: stepResult.ApplicationID,
        Data: map[string]interface{}{
            "step_type": stepResult.StepType,
            "status":    stepResult.Status,
            "decision":  stepResult.Decision,
        },
    })
    
    // Handle step result
    switch stepResult.Status {
    case "APPROVED":
        return lawm.processNextStep(ctx, stepResult.ApplicationID)
    case "REJECTED":
        return lawm.rejectApplication(ctx, stepResult.ApplicationID, stepResult.Reason)
    case "REQUIRES_MANUAL_REVIEW":
        return lawm.escalateToManualReview(ctx, stepResult.ApplicationID)
    default:
        return fmt.Errorf("unknown step status: %s", stepResult.Status)
    }
}

func (lawm *LoanApprovalWorkflowMediator) findProcessorForStep(stepType string, application *LoanApplication) LoanProcessor {
    lawm.mu.RLock()
    defer lawm.mu.RUnlock()
    
    for _, processor := range lawm.processors {
        if processor.GetProcessorType() == stepType && processor.CanProcess(application) {
            return processor
        }
    }
    
    return nil
}

func (lawm *LoanApprovalWorkflowMediator) defineWorkflowSteps(app *LoanApplication) []*WorkflowStep {
    steps := []*WorkflowStep{
        {StepType: "CREDIT_CHECK", Status: "PENDING"},
        {StepType: "INCOME_VERIFICATION", Status: "PENDING"},
        {StepType: "COLLATERAL_ASSESSMENT", Status: "PENDING"},
    }
    
    // Add conditional steps based on loan amount
    if app.RequestedAmount.GreaterThan(decimal.NewFromInt(100000)) {
        steps = append(steps, &WorkflowStep{StepType: "SENIOR_APPROVAL", Status: "PENDING"})
    }
    
    steps = append(steps, &WorkflowStep{StepType: "FINAL_APPROVAL", Status: "PENDING"})
    
    return steps
}

func (lawm *LoanApprovalWorkflowMediator) notifyProcessors(ctx context.Context, notification *LoanNotification) {
    lawm.mu.RLock()
    defer lawm.mu.RUnlock()
    
    for _, processor := range lawm.processors {
        go func(p LoanProcessor) {
            if err := p.HandleNotification(ctx, notification); err != nil {
                lawm.logger.Warn("Processor notification failed", 
                    zap.String("processor_id", p.GetProcessorID()),
                    zap.Error(err))
            }
        }(processor)
    }
}

// Concrete processors
type CreditCheckProcessor struct {
    processorID   string
    creditBureau  CreditBureau
    scoreThreshold int
    logger        *zap.Logger
}

func NewCreditCheckProcessor(processorID string, creditBureau CreditBureau, scoreThreshold int, logger *zap.Logger) *CreditCheckProcessor {
    return &CreditCheckProcessor{
        processorID:    processorID,
        creditBureau:   creditBureau,
        scoreThreshold: scoreThreshold,
        logger:         logger,
    }
}

func (ccp *CreditCheckProcessor) GetProcessorID() string {
    return ccp.processorID
}

func (ccp *CreditCheckProcessor) GetProcessorType() string {
    return "CREDIT_CHECK"
}

func (ccp *CreditCheckProcessor) CanProcess(application *LoanApplication) bool {
    return application.ApplicantSSN != ""
}

func (ccp *CreditCheckProcessor) ProcessApplication(ctx context.Context, application *LoanApplication) (*StepResult, error) {
    ccp.logger.Info("Processing credit check", 
        zap.String("application_id", application.ApplicationID))
    
    // Perform credit check
    creditReport, err := ccp.creditBureau.GetCreditReport(ctx, application.ApplicantSSN)
    if err != nil {
        return nil, fmt.Errorf("credit check failed: %w", err)
    }
    
    result := &StepResult{
        ApplicationID: application.ApplicationID,
        StepType:      "CREDIT_CHECK",
        ProcessedAt:   time.Now(),
        Data: map[string]interface{}{
            "credit_score": creditReport.Score,
            "report_date":  creditReport.Date,
        },
    }
    
    if creditReport.Score >= ccp.scoreThreshold {
        result.Status = "APPROVED"
        result.Decision = "CREDIT_APPROVED"
        result.Reason = fmt.Sprintf("Credit score %d meets threshold %d", creditReport.Score, ccp.scoreThreshold)
    } else {
        result.Status = "REJECTED"
        result.Decision = "CREDIT_REJECTED"
        result.Reason = fmt.Sprintf("Credit score %d below threshold %d", creditReport.Score, ccp.scoreThreshold)
    }
    
    return result, nil
}

func (ccp *CreditCheckProcessor) HandleNotification(ctx context.Context, notification *LoanNotification) error {
    switch notification.Type {
    case "APPLICATION_SUBMITTED":
        ccp.logger.Info("New loan application notification received", 
            zap.String("application_id", notification.ApplicationID))
    }
    return nil
}

// Income verification processor
type IncomeVerificationProcessor struct {
    processorID       string
    incomeThreshold   decimal.Decimal
    verificationAPI   IncomeVerificationAPI
    logger           *zap.Logger
}

func NewIncomeVerificationProcessor(processorID string, threshold decimal.Decimal, api IncomeVerificationAPI, logger *zap.Logger) *IncomeVerificationProcessor {
    return &IncomeVerificationProcessor{
        processorID:     processorID,
        incomeThreshold: threshold,
        verificationAPI: api,
        logger:          logger,
    }
}

func (ivp *IncomeVerificationProcessor) GetProcessorID() string {
    return ivp.processorID
}

func (ivp *IncomeVerificationProcessor) GetProcessorType() string {
    return "INCOME_VERIFICATION"
}

func (ivp *IncomeVerificationProcessor) CanProcess(application *LoanApplication) bool {
    return application.EmployerInfo != nil
}

func (ivp *IncomeVerificationProcessor) ProcessApplication(ctx context.Context, application *LoanApplication) (*StepResult, error) {
    ivp.logger.Info("Processing income verification", 
        zap.String("application_id", application.ApplicationID))
    
    // Verify income with employer or tax records
    income, err := ivp.verificationAPI.VerifyIncome(ctx, application.ApplicantSSN, application.EmployerInfo)
    if err != nil {
        return nil, fmt.Errorf("income verification failed: %w", err)
    }
    
    result := &StepResult{
        ApplicationID: application.ApplicationID,
        StepType:      "INCOME_VERIFICATION",
        ProcessedAt:   time.Now(),
        Data: map[string]interface{}{
            "verified_income":  income.MonthlyIncome,
            "employment_years": income.YearsEmployed,
        },
    }
    
    // Calculate debt-to-income ratio
    monthlyPayment := application.RequestedAmount.Div(decimal.NewFromInt(360)) // 30-year loan approximation
    debtToIncomeRatio := monthlyPayment.Div(income.MonthlyIncome)
    
    if debtToIncomeRatio.LessThanOrEqual(decimal.NewFromFloat(0.43)) { // 43% DTI threshold
        result.Status = "APPROVED"
        result.Decision = "INCOME_SUFFICIENT"
        result.Reason = fmt.Sprintf("DTI ratio %.2f%% is acceptable", debtToIncomeRatio.InexactFloat64()*100)
    } else {
        result.Status = "REJECTED"
        result.Decision = "INCOME_INSUFFICIENT"
        result.Reason = fmt.Sprintf("DTI ratio %.2f%% exceeds limit", debtToIncomeRatio.InexactFloat64()*100)
    }
    
    return result, nil
}

func (ivp *IncomeVerificationProcessor) HandleNotification(ctx context.Context, notification *LoanNotification) error {
    // Handle notifications as needed
    return nil
}
```

## Go Implementation

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
    "github.com/shopspring/decimal"
    "go.uber.org/zap"
)

// Example: Chat room mediator
// Demonstrates how mediator coordinates communication between multiple chat participants

// Participant interface
type ChatParticipant interface {
    GetID() string
    GetName() string
    SendMessage(message string) error
    ReceiveMessage(from ChatParticipant, message string) error
    SetMediator(mediator ChatMediator)
    Join() error
    Leave() error
}

// Mediator interface
type ChatMediator interface {
    AddParticipant(participant ChatParticipant) error
    RemoveParticipant(participantID string) error
    SendMessage(from ChatParticipant, message string, to ...string) error
    BroadcastMessage(from ChatParticipant, message string) error
    GetParticipants() []ChatParticipant
    GetParticipantCount() int
}

// Concrete mediator implementation
type ChatRoom struct {
    roomID       string
    participants map[string]ChatParticipant
    messageHistory []ChatMessage
    logger       *zap.Logger
    mu           sync.RWMutex
}

type ChatMessage struct {
    ID        string
    From      string
    To        []string
    Message   string
    Timestamp time.Time
    Type      string // "direct", "broadcast"
}

func NewChatRoom(roomID string, logger *zap.Logger) *ChatRoom {
    return &ChatRoom{
        roomID:         roomID,
        participants:   make(map[string]ChatParticipant),
        messageHistory: make([]ChatMessage, 0),
        logger:         logger,
    }
}

func (cr *ChatRoom) AddParticipant(participant ChatParticipant) error {
    cr.mu.Lock()
    defer cr.mu.Unlock()
    
    participantID := participant.GetID()
    if _, exists := cr.participants[participantID]; exists {
        return fmt.Errorf("participant %s already exists in room", participantID)
    }
    
    cr.participants[participantID] = participant
    participant.SetMediator(cr)
    
    cr.logger.Info("Participant joined chat room", 
        zap.String("room_id", cr.roomID),
        zap.String("participant_id", participantID),
        zap.String("participant_name", participant.GetName()))
    
    // Notify other participants
    joinMessage := fmt.Sprintf("%s joined the chat", participant.GetName())
    cr.broadcastSystemMessage(joinMessage, participantID)
    
    return nil
}

func (cr *ChatRoom) RemoveParticipant(participantID string) error {
    cr.mu.Lock()
    defer cr.mu.Unlock()
    
    participant, exists := cr.participants[participantID]
    if !exists {
        return fmt.Errorf("participant %s not found", participantID)
    }
    
    delete(cr.participants, participantID)
    
    cr.logger.Info("Participant left chat room", 
        zap.String("room_id", cr.roomID),
        zap.String("participant_id", participantID),
        zap.String("participant_name", participant.GetName()))
    
    // Notify other participants
    leaveMessage := fmt.Sprintf("%s left the chat", participant.GetName())
    cr.broadcastSystemMessage(leaveMessage, participantID)
    
    return nil
}

func (cr *ChatRoom) SendMessage(from ChatParticipant, message string, to ...string) error {
    cr.mu.RLock()
    defer cr.mu.RUnlock()
    
    fromID := from.GetID()
    
    // Verify sender is in the room
    if _, exists := cr.participants[fromID]; !exists {
        return fmt.Errorf("sender %s not in room", fromID)
    }
    
    chatMessage := ChatMessage{
        ID:        generateMessageID(),
        From:      fromID,
        To:        to,
        Message:   message,
        Timestamp: time.Now(),
        Type:      "direct",
    }
    
    if len(to) == 0 {
        // Broadcast to all
        chatMessage.Type = "broadcast"
        cr.messageHistory = append(cr.messageHistory, chatMessage)
        
        cr.logger.Debug("Broadcasting message", 
            zap.String("room_id", cr.roomID),
            zap.String("from", from.GetName()),
            zap.String("message", message))
        
        for participantID, participant := range cr.participants {
            if participantID != fromID { // Don't send to sender
                go func(p ChatParticipant) {
                    if err := p.ReceiveMessage(from, message); err != nil {
                        cr.logger.Warn("Failed to deliver message", 
                            zap.String("to", p.GetID()),
                            zap.Error(err))
                    }
                }(participant)
            }
        }
    } else {
        // Direct message to specific participants
        cr.messageHistory = append(cr.messageHistory, chatMessage)
        
        cr.logger.Debug("Sending direct message", 
            zap.String("room_id", cr.roomID),
            zap.String("from", from.GetName()),
            zap.Strings("to", to),
            zap.String("message", message))
        
        for _, recipientID := range to {
            if recipient, exists := cr.participants[recipientID]; exists {
                go func(p ChatParticipant) {
                    if err := p.ReceiveMessage(from, message); err != nil {
                        cr.logger.Warn("Failed to deliver direct message", 
                            zap.String("to", p.GetID()),
                            zap.Error(err))
                    }
                }(recipient)
            } else {
                cr.logger.Warn("Recipient not found", 
                    zap.String("recipient_id", recipientID))
            }
        }
    }
    
    return nil
}

func (cr *ChatRoom) BroadcastMessage(from ChatParticipant, message string) error {
    return cr.SendMessage(from, message) // Empty 'to' list means broadcast
}

func (cr *ChatRoom) broadcastSystemMessage(message string, excludeParticipant string) {
    systemMessage := ChatMessage{
        ID:        generateMessageID(),
        From:      "SYSTEM",
        Message:   message,
        Timestamp: time.Now(),
        Type:      "system",
    }
    
    cr.messageHistory = append(cr.messageHistory, systemMessage)
    
    for participantID, participant := range cr.participants {
        if participantID != excludeParticipant {
            go func(p ChatParticipant) {
                // Create a system participant for the message
                systemParticipant := &SystemParticipant{id: "SYSTEM", name: "System"}
                if err := p.ReceiveMessage(systemParticipant, message); err != nil {
                    cr.logger.Warn("Failed to deliver system message", 
                        zap.String("to", p.GetID()),
                        zap.Error(err))
                }
            }(participant)
        }
    }
}

func (cr *ChatRoom) GetParticipants() []ChatParticipant {
    cr.mu.RLock()
    defer cr.mu.RUnlock()
    
    participants := make([]ChatParticipant, 0, len(cr.participants))
    for _, participant := range cr.participants {
        participants = append(participants, participant)
    }
    
    return participants
}

func (cr *ChatRoom) GetParticipantCount() int {
    cr.mu.RLock()
    defer cr.mu.RUnlock()
    return len(cr.participants)
}

func (cr *ChatRoom) GetMessageHistory() []ChatMessage {
    cr.mu.RLock()
    defer cr.mu.RUnlock()
    
    history := make([]ChatMessage, len(cr.messageHistory))
    copy(history, cr.messageHistory)
    return history
}

// Concrete participant implementations
type User struct {
    id        string
    name      string
    mediator  ChatMediator
    online    bool
    logger    *zap.Logger
    mu        sync.RWMutex
}

func NewUser(id, name string, logger *zap.Logger) *User {
    return &User{
        id:     id,
        name:   name,
        online: false,
        logger: logger,
    }
}

func (u *User) GetID() string {
    return u.id
}

func (u *User) GetName() string {
    return u.name
}

func (u *User) SetMediator(mediator ChatMediator) {
    u.mu.Lock()
    defer u.mu.Unlock()
    u.mediator = mediator
}

func (u *User) Join() error {
    u.mu.Lock()
    defer u.mu.Unlock()
    
    if u.mediator == nil {
        return fmt.Errorf("no mediator set")
    }
    
    if err := u.mediator.AddParticipant(u); err != nil {
        return err
    }
    
    u.online = true
    u.logger.Info("User joined chat", 
        zap.String("user_id", u.id),
        zap.String("user_name", u.name))
    
    return nil
}

func (u *User) Leave() error {
    u.mu.Lock()
    defer u.mu.Unlock()
    
    if u.mediator == nil {
        return fmt.Errorf("no mediator set")
    }
    
    if err := u.mediator.RemoveParticipant(u.id); err != nil {
        return err
    }
    
    u.online = false
    u.logger.Info("User left chat", 
        zap.String("user_id", u.id),
        zap.String("user_name", u.name))
    
    return nil
}

func (u *User) SendMessage(message string) error {
    u.mu.RLock()
    defer u.mu.RUnlock()
    
    if !u.online || u.mediator == nil {
        return fmt.Errorf("user not online or no mediator")
    }
    
    return u.mediator.BroadcastMessage(u, message)
}

func (u *User) SendDirectMessage(message string, toUserIDs ...string) error {
    u.mu.RLock()
    defer u.mu.RUnlock()
    
    if !u.online || u.mediator == nil {
        return fmt.Errorf("user not online or no mediator")
    }
    
    return u.mediator.SendMessage(u, message, toUserIDs...)
}

func (u *User) ReceiveMessage(from ChatParticipant, message string) error {
    u.mu.RLock()
    defer u.mu.RUnlock()
    
    if !u.online {
        return fmt.Errorf("user is offline")
    }
    
    u.logger.Info("Message received", 
        zap.String("to", u.name),
        zap.String("from", from.GetName()),
        zap.String("message", message))
    
    // In a real implementation, this might display the message in UI,
    // store in local message history, trigger notifications, etc.
    fmt.Printf("[%s] %s: %s\n", u.name, from.GetName(), message)
    
    return nil
}

// Bot participant
type ChatBot struct {
    id       string
    name     string
    mediator ChatMediator
    commands map[string]func(string, ChatParticipant) string
    logger   *zap.Logger
    mu       sync.RWMutex
}

func NewChatBot(id, name string, logger *zap.Logger) *ChatBot {
    bot := &ChatBot{
        id:       id,
        name:     name,
        commands: make(map[string]func(string, ChatParticipant) string),
        logger:   logger,
    }
    
    bot.setupCommands()
    return bot
}

func (cb *ChatBot) setupCommands() {
    cb.commands["!help"] = func(args string, from ChatParticipant) string {
        return "Available commands: !help, !time, !participants"
    }
    
    cb.commands["!time"] = func(args string, from ChatParticipant) string {
        return fmt.Sprintf("Current time: %s", time.Now().Format("15:04:05"))
    }
    
    cb.commands["!participants"] = func(args string, from ChatParticipant) string {
        if cb.mediator != nil {
            count := cb.mediator.GetParticipantCount()
            return fmt.Sprintf("Participants in chat: %d", count)
        }
        return "Unable to get participant count"
    }
}

func (cb *ChatBot) GetID() string {
    return cb.id
}

func (cb *ChatBot) GetName() string {
    return cb.name
}

func (cb *ChatBot) SetMediator(mediator ChatMediator) {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    cb.mediator = mediator
}

func (cb *ChatBot) Join() error {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    
    if cb.mediator == nil {
        return fmt.Errorf("no mediator set")
    }
    
    return cb.mediator.AddParticipant(cb)
}

func (cb *ChatBot) Leave() error {
    cb.mu.Lock()
    defer cb.mu.Unlock()
    
    if cb.mediator == nil {
        return fmt.Errorf("no mediator set")
    }
    
    return cb.mediator.RemoveParticipant(cb.id)
}

func (cb *ChatBot) SendMessage(message string) error {
    cb.mu.RLock()
    defer cb.mu.RUnlock()
    
    if cb.mediator == nil {
        return fmt.Errorf("no mediator set")
    }
    
    return cb.mediator.BroadcastMessage(cb, message)
}

func (cb *ChatBot) ReceiveMessage(from ChatParticipant, message string) error {
    cb.mu.RLock()
    defer cb.mu.RUnlock()
    
    cb.logger.Debug("Bot received message", 
        zap.String("from", from.GetName()),
        zap.String("message", message))
    
    // Check if message is a command
    if len(message) > 0 && message[0] == '!' {
        cmd := message
        if handler, exists := cb.commands[cmd]; exists {
            response := handler("", from)
            
            // Send response back through mediator
            if cb.mediator != nil {
                go func() {
                    if err := cb.mediator.SendMessage(cb, response, from.GetID()); err != nil {
                        cb.logger.Error("Failed to send bot response", zap.Error(err))
                    }
                }()
            }
        }
    }
    
    return nil
}

// System participant for system messages
type SystemParticipant struct {
    id   string
    name string
}

func (sp *SystemParticipant) GetID() string {
    return sp.id
}

func (sp *SystemParticipant) GetName() string {
    return sp.name
}

func (sp *SystemParticipant) SendMessage(message string) error {
    return fmt.Errorf("system participant cannot send messages")
}

func (sp *SystemParticipant) ReceiveMessage(from ChatParticipant, message string) error {
    return fmt.Errorf("system participant cannot receive messages")
}

func (sp *SystemParticipant) SetMediator(mediator ChatMediator) {
    // No-op for system participant
}

func (sp *SystemParticipant) Join() error {
    return fmt.Errorf("system participant cannot join")
}

func (sp *SystemParticipant) Leave() error {
    return fmt.Errorf("system participant cannot leave")
}

// Helper functions
func generateMessageID() string {
    return fmt.Sprintf("msg_%d", time.Now().UnixNano())
}

// Example usage
func main() {
    fmt.Println("=== Mediator Pattern Demo ===\n")
    
    // Create logger
    logger, _ := zap.NewDevelopment()
    defer logger.Sync()
    
    // Create chat room (mediator)
    chatRoom := NewChatRoom("general", logger)
    
    // Create participants
    alice := NewUser("user1", "Alice", logger)
    bob := NewUser("user2", "Bob", logger)
    charlie := NewUser("user3", "Charlie", logger)
    bot := NewChatBot("bot1", "HelpBot", logger)
    
    // Set the mediator for all participants
    alice.SetMediator(chatRoom)
    bob.SetMediator(chatRoom)
    charlie.SetMediator(chatRoom)
    bot.SetMediator(chatRoom)
    
    // Participants join the chat
    fmt.Println("=== Participants Joining ===")
    alice.Join()
    bob.Join()
    charlie.Join()
    bot.Join()
    
    // Wait a moment for join notifications
    time.Sleep(100 * time.Millisecond)
    
    // Send some messages
    fmt.Println("\n=== Chat Messages ===")
    alice.SendMessage("Hello everyone!")
    time.Sleep(50 * time.Millisecond)
    
    bob.SendMessage("Hi Alice! How are you?")
    time.Sleep(50 * time.Millisecond)
    
    charlie.SendMessage("Good morning!")
    time.Sleep(50 * time.Millisecond)
    
    // Send direct message
    fmt.Println("\n=== Direct Messages ===")
    alice.SendDirectMessage("Bob, let's talk privately", bob.GetID())
    time.Sleep(50 * time.Millisecond)
    
    bob.SendDirectMessage("Sure Alice, what's up?", alice.GetID())
    time.Sleep(50 * time.Millisecond)
    
    // Interact with bot
    fmt.Println("\n=== Bot Interactions ===")
    alice.SendMessage("!help")
    time.Sleep(100 * time.Millisecond)
    
    bob.SendMessage("!time")
    time.Sleep(100 * time.Millisecond)
    
    charlie.SendMessage("!participants")
    time.Sleep(100 * time.Millisecond)
    
    // Someone leaves
    fmt.Println("\n=== Participant Leaving ===")
    charlie.Leave()
    time.Sleep(100 * time.Millisecond)
    
    // More messages after someone left
    alice.SendMessage("Charlie left, but we're still here!")
    time.Sleep(50 * time.Millisecond)
    
    bob.SendMessage("Yes, the conversation continues!")
    time.Sleep(50 * time.Millisecond)
    
    // Display chat statistics
    fmt.Println("\n=== Chat Statistics ===")
    fmt.Printf("Participants in room: %d\n", chatRoom.GetParticipantCount())
    
    participants := chatRoom.GetParticipants()
    fmt.Println("Current participants:")
    for _, participant := range participants {
        fmt.Printf("  - %s (%s)\n", participant.GetName(), participant.GetID())
    }
    
    messageHistory := chatRoom.GetMessageHistory()
    fmt.Printf("\nTotal messages in history: %d\n", len(messageHistory))
    
    fmt.Println("\nMessage history:")
    for i, msg := range messageHistory {
        fmt.Printf("  %d. [%s] %s: %s (Type: %s)\n", 
            i+1, 
            msg.Timestamp.Format("15:04:05"), 
            msg.From, 
            msg.Message, 
            msg.Type)
    }
    
    // Cleanup
    fmt.Println("\n=== Cleanup ===")
    alice.Leave()
    bob.Leave()
    bot.Leave()
    
    fmt.Printf("Final participant count: %d\n", chatRoom.GetParticipantCount())
    
    fmt.Println("\n=== Mediator Pattern Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Abstract Mediator**
```go
type AbstractMediator interface {
    Notify(sender Component, event string) error
}

type Component interface {
    SetMediator(mediator AbstractMediator)
    GetComponentID() string
}

type ConcreteMediator struct {
    components map[string]Component
}

func (cm *ConcreteMediator) Notify(sender Component, event string) error {
    switch event {
    case "EVENT_A":
        return cm.handleEventA(sender)
    case "EVENT_B":
        return cm.handleEventB(sender)
    }
    return nil
}
```

2. **Event-Driven Mediator**
```go
type EventMediator struct {
    eventHandlers map[string][]EventHandler
    eventQueue    chan Event
    workers       int
}

type Event struct {
    Type      string
    Source    string
    Data      interface{}
    Timestamp time.Time
}

type EventHandler func(Event) error

func (em *EventMediator) PublishEvent(event Event) error {
    select {
    case em.eventQueue <- event:
        return nil
    default:
        return fmt.Errorf("event queue full")
    }
}

func (em *EventMediator) Subscribe(eventType string, handler EventHandler) {
    em.eventHandlers[eventType] = append(em.eventHandlers[eventType], handler)
}
```

3. **Hierarchical Mediator**
```go
type HierarchicalMediator struct {
    parent   Mediator
    children []Mediator
    localComponents map[string]Component
}

func (hm *HierarchicalMediator) Notify(sender Component, event string) error {
    // Handle locally first
    if err := hm.handleLocally(sender, event); err != nil {
        return err
    }
    
    // Propagate to parent if needed
    if hm.shouldPropagateUp(event) && hm.parent != nil {
        return hm.parent.Notify(sender, event)
    }
    
    // Propagate to children if needed
    if hm.shouldPropagateDown(event) {
        for _, child := range hm.children {
            child.Notify(sender, event)
        }
    }
    
    return nil
}
```

### Trade-offs

**Pros:**
- **Loose Coupling**: Objects don't need to know about each other directly
- **Centralized Control**: Communication logic is centralized and easier to manage
- **Reusability**: Components can be reused in different contexts with different mediators
- **Flexibility**: Easy to change interaction patterns by modifying the mediator
- **Single Responsibility**: Each component focuses on its core responsibility

**Cons:**
- **Complexity**: Mediator can become complex if it handles too many interactions
- **Single Point of Failure**: Mediator becomes a critical dependency
- **Performance**: Additional indirection can impact performance
- **God Object**: Risk of mediator becoming a "god object" that knows too much
- **Debugging**: Can be harder to trace communication flow

**When to Choose Mediator vs Alternatives:**

| Scenario | Pattern | Reason |
|----------|---------|--------|
| Complex interactions | Mediator | Centralized coordination |
| Simple notifications | Observer | Direct event subscription |
| Hierarchical communication | Chain of Responsibility | Sequential handling |
| Request processing | Command | Encapsulated requests |
| State-dependent behavior | State | Behavior changes with state |

## Integration Tips

### 1. Observer Pattern Integration
```go
type ObservableMediator struct {
    *BaseMediator
    observers []MediatorObserver
}

type MediatorObserver interface {
    OnCommunication(from, to Component, message interface{})
    OnComponentAdded(component Component)
    OnComponentRemoved(componentID string)
}

func (om *ObservableMediator) Notify(sender Component, event string) error {
    // Process the event
    err := om.BaseMediator.Notify(sender, event)
    
    // Notify observers
    for _, observer := range om.observers {
        observer.OnCommunication(sender, nil, event)
    }
    
    return err
}

func (om *ObservableMediator) AddObserver(observer MediatorObserver) {
    om.observers = append(om.observers, observer)
}
```

### 2. Command Pattern Integration
```go
type CommandMediator struct {
    *BaseMediator
    commandQueue chan Command
    commandHistory []Command
}

type MediatorCommand interface {
    Execute() error
    Undo() error
    GetSender() Component
    GetEvent() string
}

func (cm *CommandMediator) QueueCommand(command MediatorCommand) error {
    select {
    case cm.commandQueue <- command:
        return nil
    default:
        return fmt.Errorf("command queue full")
    }
}

func (cm *CommandMediator) ProcessCommands() {
    for command := range cm.commandQueue {
        if err := command.Execute(); err != nil {
            cm.logger.Error("Command execution failed", zap.Error(err))
        } else {
            cm.commandHistory = append(cm.commandHistory, command)
        }
    }
}
```

### 3. State Pattern Integration
```go
type StatefulMediator struct {
    *BaseMediator
    currentState MediatorState
}

type MediatorState interface {
    HandleNotification(mediator *StatefulMediator, sender Component, event string) error
    Enter(mediator *StatefulMediator) error
    Exit(mediator *StatefulMediator) error
}

type ActiveState struct{}

func (as *ActiveState) HandleNotification(mediator *StatefulMediator, sender Component, event string) error {
    // Handle notifications in active state
    return mediator.BaseMediator.Notify(sender, event)
}

type PausedState struct{}

func (ps *PausedState) HandleNotification(mediator *StatefulMediator, sender Component, event string) error {
    // Queue or ignore notifications in paused state
    return fmt.Errorf("mediator is paused")
}

func (sm *StatefulMediator) ChangeState(newState MediatorState) error {
    if sm.currentState != nil {
        if err := sm.currentState.Exit(sm); err != nil {
            return err
        }
    }
    
    sm.currentState = newState
    return newState.Enter(sm)
}
```

### 4. Strategy Pattern Integration
```go
type CommunicationStrategy interface {
    DeliverMessage(from Component, to Component, message interface{}) error
}

type SynchronousStrategy struct{}

func (ss *SynchronousStrategy) DeliverMessage(from Component, to Component, message interface{}) error {
    return to.ReceiveMessage(from, message)
}

type AsynchronousStrategy struct {
    messageQueue chan MessageDelivery
}

type MessageDelivery struct {
    From    Component
    To      Component
    Message interface{}
}

func (as *AsynchronousStrategy) DeliverMessage(from Component, to Component, message interface{}) error {
    delivery := MessageDelivery{
        From:    from,
        To:      to,
        Message: message,
    }
    
    select {
    case as.messageQueue <- delivery:
        return nil
    default:
        return fmt.Errorf("message queue full")
    }
}

type StrategyBasedMediator struct {
    *BaseMediator
    strategy CommunicationStrategy
}

func (sbm *StrategyBasedMediator) SetStrategy(strategy CommunicationStrategy) {
    sbm.strategy = strategy
}

func (sbm *StrategyBasedMediator) DeliverMessage(from Component, to Component, message interface{}) error {
    return sbm.strategy.DeliverMessage(from, to, message)
}
```

## Common Interview Questions

### 1. **How does Mediator pattern differ from Observer pattern?**

**Answer:**
Both patterns deal with communication between objects, but they serve different purposes and have different structures:

**Mediator Pattern:**
```go
// Centralized communication through a mediator
type ChatMediator interface {
    SendMessage(from User, message string, to ...User) error
}

type User struct {
    id       string
    mediator ChatMediator
}

func (u *User) SendMessage(message string, recipients ...User) error {
    // All communication goes through the mediator
    return u.mediator.SendMessage(u, message, recipients...)
}

// Users don't know about each other directly
user1.SendMessage("Hello", user2, user3) // Mediator coordinates
```

**Observer Pattern:**
```go
// Direct subscription-based communication
type Subject interface {
    Subscribe(observer Observer)
    Unsubscribe(observer Observer)
    Notify(event Event)
}

type Observer interface {
    Update(event Event)
}

type User struct {
    id        string
    observers []Observer
}

func (u *User) SendMessage(message string) {
    event := MessageEvent{From: u.id, Message: message}
    // Direct notification to all subscribers
    for _, observer := range u.observers {
        observer.Update(event)
    }
}

// Users know about their observers directly
user1.Subscribe(user2)
user1.SendMessage("Hello") // Direct notification
```

**Key Differences:**

| Aspect | Mediator | Observer |
|--------|----------|----------|
| **Communication** | Centralized through mediator | Direct publisher-subscriber |
| **Coupling** | Components coupled to mediator | Publishers coupled to observers |
| **Control** | Mediator controls all interactions | Each subject controls its observers |
| **Complexity** | Complex mediator, simple components | Simple subjects, complex interactions |
| **Use Case** | Complex multi-way communication | One-to-many event notification |

### 2. **How do you prevent the mediator from becoming a God Object?**

**Answer:**
Several strategies can prevent mediators from becoming too complex:

**1. Decompose by Domain:**
```go
// Instead of one huge mediator
type MonolithicMediator struct {
    // Handles payments, users, orders, notifications, etc.
}

// Break into domain-specific mediators
type PaymentMediator interface {
    ProcessPayment(request PaymentRequest) error
}

type UserMediator interface {
    RegisterUser(user User) error
    AuthenticateUser(credentials Credentials) error
}

type OrderMediator interface {
    CreateOrder(order Order) error
    UpdateOrderStatus(orderID string, status string) error
}

// Coordinate between mediators if needed
type SystemCoordinator struct {
    paymentMediator PaymentMediator
    userMediator    UserMediator
    orderMediator   OrderMediator
}
```

**2. Use Event-Driven Architecture:**
```go
type EventBus interface {
    Publish(event Event) error
    Subscribe(eventType string, handler EventHandler) error
}

type PaymentProcessor struct {
    eventBus EventBus
}

func (pp *PaymentProcessor) ProcessPayment(payment Payment) error {
    // Process payment
    result := pp.process(payment)
    
    // Publish event instead of direct mediator calls
    event := PaymentProcessedEvent{
        PaymentID: payment.ID,
        Status:    result.Status,
        Amount:    payment.Amount,
    }
    
    return pp.eventBus.Publish(event)
}

// Other components subscribe to events they care about
type NotificationService struct {
    eventBus EventBus
}

func (ns *NotificationService) Initialize() {
    ns.eventBus.Subscribe("PaymentProcessed", ns.handlePaymentProcessed)
    ns.eventBus.Subscribe("UserRegistered", ns.handleUserRegistered)
}
```

**3. Use Composition and Delegation:**
```go
type CompositeMediator struct {
    validators []Validator
    processors []Processor
    notifiers  []Notifier
}

func (cm *CompositeMediator) ProcessRequest(request Request) error {
    // Delegate to appropriate handlers
    for _, validator := range cm.validators {
        if validator.CanValidate(request) {
            if err := validator.Validate(request); err != nil {
                return err
            }
        }
    }
    
    for _, processor := range cm.processors {
        if processor.CanProcess(request) {
            if err := processor.Process(request); err != nil {
                return err
            }
        }
    }
    
    for _, notifier := range cm.notifiers {
        if notifier.ShouldNotify(request) {
            go notifier.Notify(request)
        }
    }
    
    return nil
}
```

**4. Use Strategy Pattern:**
```go
type MediatorStrategy interface {
    Handle(request Request) error
}

type StrategyBasedMediator struct {
    strategies map[string]MediatorStrategy
}

func (sbm *StrategyBasedMediator) Process(request Request) error {
    strategy, exists := sbm.strategies[request.Type]
    if !exists {
        return fmt.Errorf("no strategy for request type: %s", request.Type)
    }
    
    return strategy.Handle(request)
}

// Each strategy handles specific request types
type PaymentStrategy struct{}
type UserStrategy struct{}
type OrderStrategy struct{}
```

### 3. **How do you handle error propagation in mediator patterns?**

**Answer:**
Error handling in mediators requires careful consideration of how errors should be propagated and handled:

**1. Error Aggregation:**
```go
type MediatorError struct {
    Operation string
    Errors    []error
    Context   map[string]interface{}
}

func (m *MediatorError) Error() string {
    return fmt.Sprintf("mediator operation '%s' failed with %d errors", m.Operation, len(m.Errors))
}

type ErrorAggregatingMediator struct {
    components map[string]Component
}

func (eam *ErrorAggregatingMediator) BroadcastEvent(event Event) error {
    var errors []error
    
    for componentID, component := range eam.components {
        if err := component.HandleEvent(event); err != nil {
            errors = append(errors, fmt.Errorf("component %s: %w", componentID, err))
        }
    }
    
    if len(errors) > 0 {
        return &MediatorError{
            Operation: "BroadcastEvent",
            Errors:    errors,
            Context:   map[string]interface{}{"event_type": event.Type},
        }
    }
    
    return nil
}
```

**2. Circuit Breaker Integration:**
```go
type CircuitBreakerMediator struct {
    *BaseMediator
    breakers map[string]*CircuitBreaker
}

func (cbm *CircuitBreakerMediator) NotifyComponent(componentID string, event Event) error {
    breaker, exists := cbm.breakers[componentID]
    if !exists {
        breaker = NewCircuitBreaker(componentID)
        cbm.breakers[componentID] = breaker
    }
    
    return breaker.Execute(func() error {
        component := cbm.GetComponent(componentID)
        return component.HandleEvent(event)
    })
}
```

**3. Retry with Backoff:**
```go
type RetryableMediator struct {
    *BaseMediator
    maxRetries int
    backoff    BackoffStrategy
}

func (rm *RetryableMediator) NotifyComponent(componentID string, event Event) error {
    var lastErr error
    
    for attempt := 0; attempt <= rm.maxRetries; attempt++ {
        if attempt > 0 {
            delay := rm.backoff.NextDelay(attempt)
            time.Sleep(delay)
        }
        
        component := rm.GetComponent(componentID)
        err := component.HandleEvent(event)
        if err == nil {
            return nil
        }
        
        lastErr = err
        
        // Don't retry for certain error types
        if !rm.isRetryable(err) {
            break
        }
    }
    
    return fmt.Errorf("failed after %d attempts: %w", rm.maxRetries+1, lastErr)
}
```

**4. Compensating Actions:**
```go
type CompensatingMediator struct {
    *BaseMediator
    compensations map[string]func() error
}

func (cm *CompensatingMediator) ProcessWorkflow(workflow Workflow) error {
    var completedSteps []string
    
    for _, step := range workflow.Steps {
        if err := cm.executeStep(step); err != nil {
            // Execute compensating actions for completed steps
            for i := len(completedSteps) - 1; i >= 0; i-- {
                stepID := completedSteps[i]
                if compensation, exists := cm.compensations[stepID]; exists {
                    if compErr := compensation(); compErr != nil {
                        cm.logger.Error("Compensation failed", 
                            zap.String("step", stepID),
                            zap.Error(compErr))
                    }
                }
            }
            
            return fmt.Errorf("workflow failed at step %s: %w", step.ID, err)
        }
        
        completedSteps = append(completedSteps, step.ID)
    }
    
    return nil
}
```

### 4. **How do you test mediator implementations effectively?**

**Answer:**
Testing mediators requires both unit testing of mediator logic and integration testing of component interactions:

**1. Mock Components:**
```go
type MockComponent struct {
    mock.Mock
    id string
}

func (mc *MockComponent) GetID() string {
    return mc.id
}

func (mc *MockComponent) HandleEvent(event Event) error {
    args := mc.Called(event)
    return args.Error(0)
}

func TestMediatorBroadcast(t *testing.T) {
    mediator := NewChatRoom("test", logger)
    
    // Create mock components
    comp1 := &MockComponent{id: "comp1"}
    comp2 := &MockComponent{id: "comp2"}
    
    comp1.On("HandleEvent", mock.AnythingOfType("Event")).Return(nil)
    comp2.On("HandleEvent", mock.AnythingOfType("Event")).Return(nil)
    
    // Add components to mediator
    mediator.AddComponent(comp1)
    mediator.AddComponent(comp2)
    
    // Test broadcast
    event := Event{Type: "test", Data: "test data"}
    err := mediator.BroadcastEvent(event)
    
    assert.NoError(t, err)
    comp1.AssertExpectations(t)
    comp2.AssertExpectations(t)
}
```

**2. Integration Testing:**
```go
func TestPaymentProcessingIntegration(t *testing.T) {
    // Create real components
    validator := NewPaymentValidator()
    gateway := NewMockPaymentGateway()
    notifier := NewMockNotificationService()
    
    // Create mediator
    mediator := NewPaymentMediator()
    mediator.RegisterComponent(validator)
    mediator.RegisterComponent(gateway)
    mediator.RegisterComponent(notifier)
    
    // Test complete payment flow
    payment := &PaymentRequest{
        Amount:   decimal.NewFromFloat(100.00),
        Currency: "USD",
    }
    
    result, err := mediator.ProcessPayment(context.Background(), payment)
    
    assert.NoError(t, err)
    assert.NotNil(t, result)
    assert.Equal(t, "SUCCESS", result.Status)
    
    // Verify all components were involved
    gateway.AssertCalled(t, "ProcessPayment", mock.Anything)
    notifier.AssertCalled(t, "SendNotification", mock.Anything)
}
```

**3. Event Sequence Testing:**
```go
type EventRecorder struct {
    events []Event
    mu     sync.Mutex
}

func (er *EventRecorder) RecordEvent(event Event) {
    er.mu.Lock()
    defer er.mu.Unlock()
    er.events = append(er.events, event)
}

func (er *EventRecorder) GetEvents() []Event {
    er.mu.Lock()
    defer er.mu.Unlock()
    return append([]Event(nil), er.events...)
}

func TestEventSequence(t *testing.T) {
    recorder := &EventRecorder{}
    mediator := NewObservableMediator(recorder)
    
    // Add components
    comp1 := NewTestComponent("comp1")
    comp2 := NewTestComponent("comp2")
    
    mediator.AddComponent(comp1)
    mediator.AddComponent(comp2)
    
    // Trigger sequence of events
    mediator.ProcessRequest(TestRequest{Type: "test"})
    
    // Verify event sequence
    events := recorder.GetEvents()
    expectedSequence := []string{"REQUEST_RECEIVED", "VALIDATION_STARTED", "VALIDATION_COMPLETED", "PROCESSING_STARTED", "PROCESSING_COMPLETED"}
    
    assert.Equal(t, len(expectedSequence), len(events))
    for i, expected := range expectedSequence {
        assert.Equal(t, expected, events[i].Type)
    }
}
```

**4. Error Scenario Testing:**
```go
func TestMediatorErrorHandling(t *testing.T) {
    mediator := NewPaymentMediator()
    
    // Create component that always fails
    failingComponent := &MockComponent{id: "failing"}
    failingComponent.On("HandleEvent", mock.Anything).Return(fmt.Errorf("component failure"))
    
    // Create normal component
    normalComponent := &MockComponent{id: "normal"}
    normalComponent.On("HandleEvent", mock.Anything).Return(nil)
    
    mediator.AddComponent(failingComponent)
    mediator.AddComponent(normalComponent)
    
    // Test error handling
    event := Event{Type: "test"}
    err := mediator.BroadcastEvent(event)
    
    // Should return error but not affect other components
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "component failure")
    
    // Normal component should still have been called
    normalComponent.AssertCalled(t, "HandleEvent", event)
}
```

### 5. **When should you avoid using Mediator pattern?**

**Answer:**
Mediator pattern should be avoided in certain scenarios:

**1. Simple One-to-One Communication:**
```go
// DON'T use mediator for simple direct communication
type UserService struct {
    emailService EmailService
}

func (us *UserService) CreateUser(user User) error {
    if err := us.validateUser(user); err != nil {
        return err
    }
    
    if err := us.saveUser(user); err != nil {
        return err
    }
    
    // Direct call is simpler than mediator
    return us.emailService.SendWelcomeEmail(user.Email)
}

// Instead of unnecessary mediator:
// mediator.Notify(userService, "USER_CREATED", user)
```

**2. Performance-Critical Systems:**
```go
// DON'T use mediator in high-frequency trading systems
type HighFrequencyTrader struct {
    orderBook OrderBook
    riskCheck RiskChecker
}

func (hft *HighFrequencyTrader) PlaceOrder(order Order) error {
    // Direct calls for microsecond-sensitive operations
    if !hft.riskCheck.IsAllowed(order) {
        return errors.New("risk check failed")
    }
    
    return hft.orderBook.AddOrder(order)
    
    // Mediator would add unnecessary latency:
    // return mediator.ProcessOrder(order) // Too slow
}
```

**3. Simple Event Systems:**
```go
// DON'T use mediator for simple pub/sub
type SimplePublisher struct {
    subscribers []Subscriber
}

func (sp *SimplePublisher) Publish(event Event) {
    // Direct notification is simpler
    for _, subscriber := range sp.subscribers {
        go subscriber.Handle(event)
    }
}

// No need for mediator when Observer pattern is sufficient
```

**4. Hierarchical Systems:**
```go
// DON'T use mediator for natural hierarchies
type OrderProcessor struct {
    paymentProcessor PaymentProcessor
    inventoryManager InventoryManager
    shippingService  ShippingService
}

func (op *OrderProcessor) ProcessOrder(order Order) error {
    // Natural workflow - no mediator needed
    if err := op.paymentProcessor.ProcessPayment(order.Payment); err != nil {
        return err
    }
    
    if err := op.inventoryManager.ReserveItems(order.Items); err != nil {
        return err
    }
    
    return op.shippingService.CreateShipment(order)
}
```

**Better Alternatives:**

| Scenario | Alternative | Reason |
|----------|-------------|--------|
| Simple notifications | Observer | Direct pub/sub |
| Sequential processing | Chain of Responsibility | Linear workflow |
| Hierarchical communication | Direct calls | Natural hierarchy |
| Event streaming | Event Bus/Message Queue | Better for high volume |
| Request-response | Direct service calls | Lower latency |
| State management | State Machine | Better state modeling |

**Decision Framework:**
```go
type MediatorDecision struct {
    ComponentCount      int
    InteractionComplexity string // "simple", "medium", "complex"
    CommunicationPattern string // "one-to-one", "one-to-many", "many-to-many"
    PerformanceNeeds    string // "low", "medium", "high"
    CouplingTolerance   string // "tight", "loose"
    SystemSize          string // "small", "medium", "large"
}

func (md *MediatorDecision) ShouldUseMediator() (bool, string) {
    if md.ComponentCount < 3 {
        return false, "Too few components to benefit from mediator"
    }
    
    if md.InteractionComplexity == "simple" && md.CommunicationPattern == "one-to-one" {
        return false, "Simple direct communication is better"
    }
    
    if md.PerformanceNeeds == "high" && md.SystemSize == "small" {
        return false, "Performance overhead not justified for small systems"
    }
    
    if md.CommunicationPattern == "many-to-many" && md.InteractionComplexity == "complex" {
        return true, "Mediator helps manage complex many-to-many interactions"
    }
    
    if md.CouplingTolerance == "loose" && md.ComponentCount > 5 {
        return true, "Mediator reduces coupling in larger systems"
    }
    
    return false, "Direct communication likely sufficient"
}
```
