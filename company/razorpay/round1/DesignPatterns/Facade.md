# Facade Pattern

## Pattern Name & Intent

**Facade** is a structural design pattern that provides a simplified interface to a complex subsystem. It defines a higher-level interface that makes the subsystem easier to use by hiding the complexity of the underlying components.

**Key Intent:**

- Provide a simple interface to a complex subsystem
- Hide the complexity of subsystem interactions
- Decouple clients from subsystem components
- Create a single entry point for subsystem functionality
- Improve subsystem usability and maintainability

## When to Use

**Use Facade when:**

1. **Complex Subsystems**: You have a complex subsystem with many components
2. **Simplified Interface**: Want to provide a simpler interface to clients
3. **Decoupling**: Need to decouple clients from implementation details
4. **Legacy Integration**: Wrapping legacy systems with modern interfaces
5. **Layered Architecture**: Creating clean separation between layers
6. **API Design**: Providing high-level APIs for complex operations
7. **Microservices**: Creating unified interfaces for multiple services

**Don't use when:**

- The subsystem is already simple
- Clients need direct access to subsystem components
- The facade would just add unnecessary indirection
- You need fine-grained control over subsystem behavior

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Processing Facade

```go
// Complex subsystem components
type PaymentValidator interface {
    ValidateCard(card *CreditCard) error
    ValidateMerchant(merchantID string) error
    ValidateAmount(amount decimal.Decimal) error
}

type FraudDetector interface {
    CheckTransaction(transaction *Transaction) (*RiskAssessment, error)
}

type PaymentGateway interface {
    ProcessPayment(payment *PaymentRequest) (*PaymentResponse, error)
}

type NotificationService interface {
    SendReceipt(email string, receipt *Receipt) error
    SendSMSConfirmation(phone string, transactionID string) error
}

type AuditLogger interface {
    LogTransaction(transaction *Transaction) error
    LogSecurityEvent(event *SecurityEvent) error
}

type InventoryService interface {
    ReserveItems(items []OrderItem) error
    ReleaseReservation(reservationID string) error
}

// Facade - simplifies the complex payment flow
type PaymentFacade struct {
    validator     PaymentValidator
    fraudDetector FraudDetector
    gateway       PaymentGateway
    notifications NotificationService
    auditLogger   AuditLogger
    inventory     InventoryService
}

func NewPaymentFacade(
    validator PaymentValidator,
    fraudDetector FraudDetector,
    gateway PaymentGateway,
    notifications NotificationService,
    auditLogger AuditLogger,
    inventory InventoryService,
) *PaymentFacade {
    return &PaymentFacade{
        validator:     validator,
        fraudDetector: fraudDetector,
        gateway:       gateway,
        notifications: notifications,
        auditLogger:   auditLogger,
        inventory:     inventory,
    }
}

// Simplified interface for complex payment processing
func (pf *PaymentFacade) ProcessPayment(request *PaymentProcessingRequest) (*PaymentResult, error) {
    // Start transaction logging
    transaction := &Transaction{
        ID:        generateTransactionID(),
        Amount:    request.Amount,
        Card:      request.Card,
        Merchant:  request.MerchantID,
        Items:     request.Items,
        Timestamp: time.Now(),
    }

    // Step 1: Validate all inputs
    if err := pf.validatePayment(request); err != nil {
        pf.auditLogger.LogSecurityEvent(&SecurityEvent{
            Type:    "VALIDATION_FAILED",
            Details: err.Error(),
        })
        return nil, fmt.Errorf("validation failed: %w", err)
    }

    // Step 2: Reserve inventory
    reservationID, err := pf.reserveInventory(request.Items)
    if err != nil {
        return nil, fmt.Errorf("inventory reservation failed: %w", err)
    }

    defer func() {
        if err != nil {
            // Release reservation if payment fails
            pf.inventory.ReleaseReservation(reservationID)
        }
    }()

    // Step 3: Fraud detection
    riskAssessment, err := pf.fraudDetector.CheckTransaction(transaction)
    if err != nil {
        return nil, fmt.Errorf("fraud detection failed: %w", err)
    }

    if riskAssessment.RiskLevel == "HIGH" {
        pf.auditLogger.LogSecurityEvent(&SecurityEvent{
            Type:         "HIGH_RISK_TRANSACTION",
            TransactionID: transaction.ID,
            RiskScore:    riskAssessment.Score,
        })
        return nil, fmt.Errorf("transaction blocked due to high risk")
    }

    // Step 4: Process payment through gateway
    paymentRequest := &PaymentRequest{
        Amount:        request.Amount,
        Card:          request.Card,
        TransactionID: transaction.ID,
        MerchantID:    request.MerchantID,
    }

    paymentResponse, err := pf.gateway.ProcessPayment(paymentRequest)
    if err != nil {
        pf.auditLogger.LogTransaction(transaction)
        return nil, fmt.Errorf("payment processing failed: %w", err)
    }

    // Step 5: Log successful transaction
    transaction.GatewayTransactionID = paymentResponse.TransactionID
    transaction.Status = "COMPLETED"
    pf.auditLogger.LogTransaction(transaction)

    // Step 6: Send notifications
    if err := pf.sendNotifications(request, paymentResponse); err != nil {
        // Don't fail the payment for notification errors, just log
        fmt.Printf("Warning: Failed to send notifications: %v\n", err)
    }

    return &PaymentResult{
        TransactionID:        transaction.ID,
        GatewayTransactionID: paymentResponse.TransactionID,
        Status:              "SUCCESS",
        Amount:              request.Amount,
        ReservationID:       reservationID,
        RiskScore:           riskAssessment.Score,
    }, nil
}

func (pf *PaymentFacade) validatePayment(request *PaymentProcessingRequest) error {
    if err := pf.validator.ValidateCard(request.Card); err != nil {
        return err
    }

    if err := pf.validator.ValidateMerchant(request.MerchantID); err != nil {
        return err
    }

    if err := pf.validator.ValidateAmount(request.Amount); err != nil {
        return err
    }

    return nil
}

func (pf *PaymentFacade) reserveInventory(items []OrderItem) (string, error) {
    if len(items) == 0 {
        return "", nil
    }

    return "", pf.inventory.ReserveItems(items)
}

func (pf *PaymentFacade) sendNotifications(request *PaymentProcessingRequest, response *PaymentResponse) error {
    // Send email receipt
    receipt := &Receipt{
        TransactionID: response.TransactionID,
        Amount:        request.Amount,
        Items:         request.Items,
        Timestamp:     time.Now(),
    }

    if request.Email != "" {
        if err := pf.notifications.SendReceipt(request.Email, receipt); err != nil {
            return err
        }
    }

    // Send SMS confirmation
    if request.Phone != "" {
        if err := pf.notifications.SendSMSConfirmation(request.Phone, response.TransactionID); err != nil {
            return err
        }
    }

    return nil
}

// Additional convenience methods
func (pf *PaymentFacade) RefundPayment(transactionID string, amount decimal.Decimal) (*RefundResult, error) {
    // Simplified refund process
    // 1. Validate refund eligibility
    // 2. Process refund through gateway
    // 3. Update inventory
    // 4. Send notifications
    // 5. Log transaction

    // Implementation details hidden from client
    return &RefundResult{
        RefundID: generateRefundID(),
        Status:   "PROCESSED",
        Amount:   amount,
    }, nil
}

func (pf *PaymentFacade) GetTransactionStatus(transactionID string) (*TransactionStatus, error) {
    // Query multiple systems and aggregate status
    return &TransactionStatus{
        TransactionID: transactionID,
        Status:        "COMPLETED",
        LastUpdated:   time.Now(),
    }, nil
}
```

### 2. Banking Operations Facade

```go
// Complex banking subsystems
type AccountService interface {
    GetAccount(accountID string) (*Account, error)
    UpdateBalance(accountID string, amount decimal.Decimal) error
    FreezeAccount(accountID string) error
}

type TransactionHistoryService interface {
    RecordTransaction(transaction *BankTransaction) error
    GetTransactionHistory(accountID string, limit int) ([]*BankTransaction, error)
}

type ComplianceService interface {
    CheckAMLCompliance(transaction *BankTransaction) (*ComplianceResult, error)
    ReportSuspiciousActivity(activity *SuspiciousActivity) error
}

type ExchangeRateService interface {
    GetExchangeRate(fromCurrency, toCurrency string) (decimal.Decimal, error)
    ConvertAmount(amount decimal.Decimal, fromCurrency, toCurrency string) (decimal.Decimal, error)
}

// Banking facade for money transfers
type BankingFacade struct {
    accountService      AccountService
    transactionHistory  TransactionHistoryService
    complianceService   ComplianceService
    exchangeRateService ExchangeRateService
    logger             *zap.Logger
}

func NewBankingFacade(
    accountService AccountService,
    transactionHistory TransactionHistoryService,
    complianceService ComplianceService,
    exchangeRateService ExchangeRateService,
    logger *zap.Logger,
) *BankingFacade {
    return &BankingFacade{
        accountService:      accountService,
        transactionHistory:  transactionHistory,
        complianceService:   complianceService,
        exchangeRateService: exchangeRateService,
        logger:             logger,
    }
}

// Simplified money transfer operation
func (bf *BankingFacade) TransferMoney(request *TransferRequest) (*TransferResult, error) {
    bf.logger.Info("Starting money transfer",
        zap.String("from", request.FromAccountID),
        zap.String("to", request.ToAccountID),
        zap.String("amount", request.Amount.String()))

    // Step 1: Validate accounts
    fromAccount, err := bf.accountService.GetAccount(request.FromAccountID)
    if err != nil {
        return nil, fmt.Errorf("invalid from account: %w", err)
    }

    toAccount, err := bf.accountService.GetAccount(request.ToAccountID)
    if err != nil {
        return nil, fmt.Errorf("invalid to account: %w", err)
    }

    // Step 2: Currency conversion if needed
    transferAmount := request.Amount
    if fromAccount.Currency != toAccount.Currency {
        convertedAmount, err := bf.exchangeRateService.ConvertAmount(
            request.Amount,
            fromAccount.Currency,
            toAccount.Currency,
        )
        if err != nil {
            return nil, fmt.Errorf("currency conversion failed: %w", err)
        }
        transferAmount = convertedAmount
    }

    // Step 3: Create transaction for compliance check
    transaction := &BankTransaction{
        ID:            generateTransactionID(),
        FromAccountID: request.FromAccountID,
        ToAccountID:   request.ToAccountID,
        Amount:        request.Amount,
        Currency:      fromAccount.Currency,
        Type:          "TRANSFER",
        Timestamp:     time.Now(),
        Description:   request.Description,
    }

    // Step 4: Compliance check
    complianceResult, err := bf.complianceService.CheckAMLCompliance(transaction)
    if err != nil {
        return nil, fmt.Errorf("compliance check failed: %w", err)
    }

    if complianceResult.Status == "BLOCKED" {
        bf.complianceService.ReportSuspiciousActivity(&SuspiciousActivity{
            TransactionID: transaction.ID,
            Reason:        complianceResult.Reason,
            RiskScore:     complianceResult.RiskScore,
        })
        return nil, fmt.Errorf("transfer blocked by compliance: %s", complianceResult.Reason)
    }

    // Step 5: Check sufficient balance
    if fromAccount.Balance.LessThan(request.Amount) {
        return nil, fmt.Errorf("insufficient balance")
    }

    // Step 6: Perform the transfer (atomic operation)
    if err := bf.performTransfer(fromAccount, toAccount, request.Amount, transferAmount); err != nil {
        return nil, fmt.Errorf("transfer failed: %w", err)
    }

    // Step 7: Record transaction
    transaction.Status = "COMPLETED"
    if err := bf.transactionHistory.RecordTransaction(transaction); err != nil {
        bf.logger.Error("Failed to record transaction", zap.Error(err))
        // Don't fail the transfer for logging errors
    }

    bf.logger.Info("Money transfer completed successfully",
        zap.String("transaction_id", transaction.ID))

    return &TransferResult{
        TransactionID:    transaction.ID,
        Status:          "SUCCESS",
        FromAccountID:   request.FromAccountID,
        ToAccountID:     request.ToAccountID,
        Amount:          request.Amount,
        TransferredAmount: transferAmount,
        ExchangeRate:    transferAmount.Div(request.Amount),
    }, nil
}

func (bf *BankingFacade) performTransfer(fromAccount, toAccount *Account, debitAmount, creditAmount decimal.Decimal) error {
    // Debit from source account
    if err := bf.accountService.UpdateBalance(fromAccount.ID, debitAmount.Neg()); err != nil {
        return fmt.Errorf("failed to debit from account: %w", err)
    }

    // Credit to destination account
    if err := bf.accountService.UpdateBalance(toAccount.ID, creditAmount); err != nil {
        // Rollback the debit
        bf.accountService.UpdateBalance(fromAccount.ID, debitAmount)
        return fmt.Errorf("failed to credit to account: %w", err)
    }

    return nil
}

// Simplified account operations
func (bf *BankingFacade) GetAccountSummary(accountID string) (*AccountSummary, error) {
    account, err := bf.accountService.GetAccount(accountID)
    if err != nil {
        return nil, err
    }

    recentTransactions, err := bf.transactionHistory.GetTransactionHistory(accountID, 10)
    if err != nil {
        bf.logger.Warn("Failed to get transaction history", zap.Error(err))
        recentTransactions = []*BankTransaction{} // Continue without transaction history
    }

    return &AccountSummary{
        Account:            account,
        RecentTransactions: recentTransactions,
        LastUpdated:        time.Now(),
    }, nil
}
```

### 3. Investment Portfolio Facade

```go
// Complex investment subsystems
type MarketDataService interface {
    GetCurrentPrice(symbol string) (decimal.Decimal, error)
    GetHistoricalPrices(symbol string, days int) ([]PriceData, error)
}

type TradingService interface {
    PlaceOrder(order *TradeOrder) (*OrderResult, error)
    CancelOrder(orderID string) error
    GetOrderStatus(orderID string) (*OrderStatus, error)
}

type PortfolioService interface {
    GetHoldings(accountID string) ([]*Holding, error)
    UpdateHolding(accountID, symbol string, quantity decimal.Decimal) error
}

type RiskAnalysisService interface {
    CalculatePortfolioRisk(holdings []*Holding) (*RiskMetrics, error)
    GetRecommendations(profile *RiskProfile) ([]*Investment, error)
}

// Investment facade
type InvestmentFacade struct {
    marketData   MarketDataService
    trading      TradingService
    portfolio    PortfolioService
    riskAnalysis RiskAnalysisService
    logger       *zap.Logger
}

func NewInvestmentFacade(
    marketData MarketDataService,
    trading TradingService,
    portfolio PortfolioService,
    riskAnalysis RiskAnalysisService,
    logger *zap.Logger,
) *InvestmentFacade {
    return &InvestmentFacade{
        marketData:   marketData,
        trading:      trading,
        portfolio:    portfolio,
        riskAnalysis: riskAnalysis,
        logger:       logger,
    }
}

// Simplified investment operation
func (inf *InvestmentFacade) InvestInPortfolio(request *InvestmentRequest) (*InvestmentResult, error) {
    // Step 1: Get current holdings
    holdings, err := inf.portfolio.GetHoldings(request.AccountID)
    if err != nil {
        return nil, fmt.Errorf("failed to get holdings: %w", err)
    }

    // Step 2: Calculate risk metrics
    currentRisk, err := inf.riskAnalysis.CalculatePortfolioRisk(holdings)
    if err != nil {
        return nil, fmt.Errorf("risk calculation failed: %w", err)
    }

    // Step 3: Get investment recommendations
    recommendations, err := inf.riskAnalysis.GetRecommendations(request.RiskProfile)
    if err != nil {
        return nil, fmt.Errorf("failed to get recommendations: %w", err)
    }

    // Step 4: Allocate investment amount across recommendations
    allocations := inf.calculateAllocations(request.Amount, recommendations)

    var orders []*OrderResult
    var totalInvested decimal.Decimal

    // Step 5: Place orders for each allocation
    for symbol, amount := range allocations {
        currentPrice, err := inf.marketData.GetCurrentPrice(symbol)
        if err != nil {
            inf.logger.Warn("Failed to get price for symbol",
                zap.String("symbol", symbol), zap.Error(err))
            continue
        }

        quantity := amount.Div(currentPrice)

        order := &TradeOrder{
            AccountID: request.AccountID,
            Symbol:    symbol,
            Quantity:  quantity,
            OrderType: "MARKET",
            Side:      "BUY",
        }

        orderResult, err := inf.trading.PlaceOrder(order)
        if err != nil {
            inf.logger.Error("Failed to place order",
                zap.String("symbol", symbol), zap.Error(err))
            continue
        }

        orders = append(orders, orderResult)
        totalInvested = totalInvested.Add(amount)

        // Update portfolio holdings
        inf.portfolio.UpdateHolding(request.AccountID, symbol, quantity)
    }

    return &InvestmentResult{
        AccountID:     request.AccountID,
        TotalInvested: totalInvested,
        Orders:        orders,
        RiskMetrics:   currentRisk,
        Timestamp:     time.Now(),
    }, nil
}

func (inf *InvestmentFacade) calculateAllocations(totalAmount decimal.Decimal, recommendations []*Investment) map[string]decimal.Decimal {
    allocations := make(map[string]decimal.Decimal)

    // Simple equal weight allocation
    weightPerRecommendation := decimal.NewFromFloat(1.0).Div(decimal.NewFromInt(int64(len(recommendations))))

    for _, rec := range recommendations {
        allocation := totalAmount.Mul(weightPerRecommendation)
        allocations[rec.Symbol] = allocation
    }

    return allocations
}

// Simplified portfolio analysis
func (inf *InvestmentFacade) GetPortfolioAnalysis(accountID string) (*PortfolioAnalysis, error) {
    // Get current holdings
    holdings, err := inf.portfolio.GetHoldings(accountID)
    if err != nil {
        return nil, err
    }

    // Calculate current values
    var totalValue decimal.Decimal
    valuedHoldings := make([]*ValuedHolding, 0, len(holdings))

    for _, holding := range holdings {
        currentPrice, err := inf.marketData.GetCurrentPrice(holding.Symbol)
        if err != nil {
            inf.logger.Warn("Failed to get current price",
                zap.String("symbol", holding.Symbol))
            continue
        }

        currentValue := holding.Quantity.Mul(currentPrice)
        totalValue = totalValue.Add(currentValue)

        valuedHoldings = append(valuedHoldings, &ValuedHolding{
            Holding:      holding,
            CurrentPrice: currentPrice,
            CurrentValue: currentValue,
        })
    }

    // Calculate risk metrics
    riskMetrics, err := inf.riskAnalysis.CalculatePortfolioRisk(holdings)
    if err != nil {
        inf.logger.Warn("Failed to calculate risk metrics", zap.Error(err))
        riskMetrics = &RiskMetrics{} // Provide empty metrics
    }

    return &PortfolioAnalysis{
        AccountID:      accountID,
        Holdings:       valuedHoldings,
        TotalValue:     totalValue,
        RiskMetrics:    riskMetrics,
        LastUpdated:    time.Now(),
    }, nil
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

// Example: E-commerce Order Processing Facade
// This facade simplifies the complex process of handling an e-commerce order

// Subsystem interfaces
type InventoryService interface {
    CheckInventory(productID string, quantity int) (bool, error)
    ReserveItems(items []OrderItem) (string, error)
    ReleaseReservation(reservationID string) error
}

type PaymentService interface {
    ProcessPayment(payment *PaymentDetails) (*PaymentResult, error)
    RefundPayment(transactionID string, amount decimal.Decimal) error
}

type ShippingService interface {
    CalculateShipping(address *Address, items []OrderItem) (*ShippingQuote, error)
    ScheduleDelivery(order *Order) (*DeliverySchedule, error)
}

type NotificationService interface {
    SendOrderConfirmation(email string, order *Order) error
    SendShippingNotification(email string, trackingNumber string) error
}

type TaxService interface {
    CalculateTax(order *Order) (decimal.Decimal, error)
}

type CustomerService interface {
    GetCustomer(customerID string) (*Customer, error)
    UpdateLoyaltyPoints(customerID string, points int) error
}

// Complex data structures
type OrderItem struct {
    ProductID string
    Name      string
    Quantity  int
    Price     decimal.Decimal
}

type Address struct {
    Street     string
    City       string
    State      string
    ZipCode    string
    Country    string
}

type PaymentDetails struct {
    CardNumber     string
    ExpiryMonth    int
    ExpiryYear     int
    CVV            string
    Amount         decimal.Decimal
    CustomerID     string
}

type Order struct {
    ID                string
    CustomerID        string
    Items             []OrderItem
    ShippingAddress   *Address
    BillingAddress    *Address
    PaymentDetails    *PaymentDetails
    Subtotal          decimal.Decimal
    Tax               decimal.Decimal
    ShippingCost      decimal.Decimal
    Total             decimal.Decimal
    Status            string
    CreatedAt         time.Time
    ReservationID     string
    TransactionID     string
    TrackingNumber    string
}

// Facade simplifies the complex order processing
type EcommerceOrderFacade struct {
    inventory     InventoryService
    payment       PaymentService
    shipping      ShippingService
    notifications NotificationService
    tax           TaxService
    customer      CustomerService
    logger        *zap.Logger
    mu            sync.RWMutex
}

func NewEcommerceOrderFacade(
    inventory InventoryService,
    payment PaymentService,
    shipping ShippingService,
    notifications NotificationService,
    tax TaxService,
    customer CustomerService,
    logger *zap.Logger,
) *EcommerceOrderFacade {
    return &EcommerceOrderFacade{
        inventory:     inventory,
        payment:       payment,
        shipping:      shipping,
        notifications: notifications,
        tax:           tax,
        customer:      customer,
        logger:        logger,
    }
}

// Simplified order processing - hides all the complexity
func (eof *EcommerceOrderFacade) PlaceOrder(ctx context.Context, orderRequest *OrderRequest) (*OrderResult, error) {
    eof.logger.Info("Starting order placement",
        zap.String("customer_id", orderRequest.CustomerID),
        zap.Int("item_count", len(orderRequest.Items)))

    // Create order with generated ID
    order := &Order{
        ID:              generateOrderID(),
        CustomerID:      orderRequest.CustomerID,
        Items:           orderRequest.Items,
        ShippingAddress: orderRequest.ShippingAddress,
        BillingAddress:  orderRequest.BillingAddress,
        PaymentDetails:  orderRequest.PaymentDetails,
        Status:          "PROCESSING",
        CreatedAt:       time.Now(),
    }

    // Step 1: Validate customer
    customer, err := eof.customer.GetCustomer(orderRequest.CustomerID)
    if err != nil {
        return nil, fmt.Errorf("customer validation failed: %w", err)
    }

    // Step 2: Check inventory and reserve items
    reservationID, err := eof.reserveInventory(order.Items)
    if err != nil {
        return nil, fmt.Errorf("inventory reservation failed: %w", err)
    }
    order.ReservationID = reservationID

    // Ensure we release reservation if anything fails
    defer func() {
        if err != nil && reservationID != "" {
            eof.inventory.ReleaseReservation(reservationID)
        }
    }()

    // Step 3: Calculate totals
    if err := eof.calculateOrderTotals(order); err != nil {
        return nil, fmt.Errorf("order calculation failed: %w", err)
    }

    // Step 4: Process payment
    paymentResult, err := eof.payment.ProcessPayment(&PaymentDetails{
        CardNumber:  orderRequest.PaymentDetails.CardNumber,
        ExpiryMonth: orderRequest.PaymentDetails.ExpiryMonth,
        ExpiryYear:  orderRequest.PaymentDetails.ExpiryYear,
        CVV:         orderRequest.PaymentDetails.CVV,
        Amount:      order.Total,
        CustomerID:  orderRequest.CustomerID,
    })
    if err != nil {
        return nil, fmt.Errorf("payment processing failed: %w", err)
    }
    order.TransactionID = paymentResult.TransactionID

    // Step 5: Schedule shipping
    deliverySchedule, err := eof.shipping.ScheduleDelivery(order)
    if err != nil {
        // Try to refund payment if shipping fails
        eof.payment.RefundPayment(paymentResult.TransactionID, order.Total)
        return nil, fmt.Errorf("shipping scheduling failed: %w", err)
    }
    order.TrackingNumber = deliverySchedule.TrackingNumber

    // Step 6: Update order status
    order.Status = "CONFIRMED"

    // Step 7: Update customer loyalty points
    loyaltyPoints := int(order.Total.InexactFloat64() / 10) // 1 point per $10
    if err := eof.customer.UpdateLoyaltyPoints(customer.ID, loyaltyPoints); err != nil {
        eof.logger.Warn("Failed to update loyalty points", zap.Error(err))
        // Don't fail the order for loyalty points
    }

    // Step 8: Send notifications
    if err := eof.sendOrderNotifications(customer.Email, order); err != nil {
        eof.logger.Warn("Failed to send notifications", zap.Error(err))
        // Don't fail the order for notification issues
    }

    eof.logger.Info("Order placed successfully",
        zap.String("order_id", order.ID),
        zap.String("transaction_id", order.TransactionID))

    return &OrderResult{
        OrderID:        order.ID,
        Status:         order.Status,
        Total:          order.Total,
        TransactionID:  order.TransactionID,
        TrackingNumber: order.TrackingNumber,
        EstimatedDelivery: deliverySchedule.EstimatedDelivery,
        LoyaltyPointsEarned: loyaltyPoints,
    }, nil
}

func (eof *EcommerceOrderFacade) reserveInventory(items []OrderItem) (string, error) {
    // Check inventory for all items first
    for _, item := range items {
        available, err := eof.inventory.CheckInventory(item.ProductID, item.Quantity)
        if err != nil {
            return "", fmt.Errorf("inventory check failed for %s: %w", item.ProductID, err)
        }
        if !available {
            return "", fmt.Errorf("insufficient inventory for product %s", item.ProductID)
        }
    }

    // Reserve all items
    reservationID, err := eof.inventory.ReserveItems(items)
    if err != nil {
        return "", fmt.Errorf("inventory reservation failed: %w", err)
    }

    return reservationID, nil
}

func (eof *EcommerceOrderFacade) calculateOrderTotals(order *Order) error {
    // Calculate subtotal
    subtotal := decimal.Zero
    for _, item := range order.Items {
        itemTotal := item.Price.Mul(decimal.NewFromInt(int64(item.Quantity)))
        subtotal = subtotal.Add(itemTotal)
    }
    order.Subtotal = subtotal

    // Calculate shipping
    shippingQuote, err := eof.shipping.CalculateShipping(order.ShippingAddress, order.Items)
    if err != nil {
        return fmt.Errorf("shipping calculation failed: %w", err)
    }
    order.ShippingCost = shippingQuote.Cost

    // Calculate tax
    tax, err := eof.tax.CalculateTax(order)
    if err != nil {
        return fmt.Errorf("tax calculation failed: %w", err)
    }
    order.Tax = tax

    // Calculate total
    order.Total = subtotal.Add(order.ShippingCost).Add(order.Tax)

    return nil
}

func (eof *EcommerceOrderFacade) sendOrderNotifications(email string, order *Order) error {
    // Send order confirmation
    if err := eof.notifications.SendOrderConfirmation(email, order); err != nil {
        return fmt.Errorf("order confirmation failed: %w", err)
    }

    // Send shipping notification if tracking number is available
    if order.TrackingNumber != "" {
        if err := eof.notifications.SendShippingNotification(email, order.TrackingNumber); err != nil {
            return fmt.Errorf("shipping notification failed: %w", err)
        }
    }

    return nil
}

// Additional simplified operations
func (eof *EcommerceOrderFacade) CancelOrder(orderID string) (*CancellationResult, error) {
    eof.logger.Info("Cancelling order", zap.String("order_id", orderID))

    // In a real implementation, this would:
    // 1. Retrieve order details
    // 2. Check if cancellation is allowed
    // 3. Release inventory reservation
    // 4. Process refund
    // 5. Cancel shipping
    // 6. Send cancellation notification
    // 7. Update order status

    return &CancellationResult{
        OrderID:       orderID,
        Status:        "CANCELLED",
        RefundAmount:  decimal.NewFromFloat(100.50), // Example amount
        RefundMethod:  "ORIGINAL_PAYMENT_METHOD",
        ProcessedAt:   time.Now(),
    }, nil
}

func (eof *EcommerceOrderFacade) GetOrderStatus(orderID string) (*OrderStatus, error) {
    // In a real implementation, this would aggregate data from multiple subsystems
    return &OrderStatus{
        OrderID:           orderID,
        Status:           "SHIPPED",
        TrackingNumber:   "TRK123456789",
        EstimatedDelivery: time.Now().Add(2 * 24 * time.Hour),
        LastUpdated:      time.Now(),
    }, nil
}

func (eof *EcommerceOrderFacade) ProcessReturn(returnRequest *ReturnRequest) (*ReturnResult, error) {
    eof.logger.Info("Processing return",
        zap.String("order_id", returnRequest.OrderID),
        zap.String("reason", returnRequest.Reason))

    // Complex return process simplified:
    // 1. Validate return eligibility
    // 2. Generate return authorization
    // 3. Schedule pickup/return shipping
    // 4. Process refund (partial or full)
    // 5. Update inventory
    // 6. Send return confirmation

    return &ReturnResult{
        ReturnID:           generateReturnID(),
        Status:            "APPROVED",
        RefundAmount:      returnRequest.RefundAmount,
        ReturnShippingLabel: "https://shipping.com/label/12345",
        ProcessedAt:       time.Now(),
    }, nil
}

// Request/Response types
type OrderRequest struct {
    CustomerID      string
    Items           []OrderItem
    ShippingAddress *Address
    BillingAddress  *Address
    PaymentDetails  *PaymentDetails
}

type OrderResult struct {
    OrderID            string
    Status             string
    Total              decimal.Decimal
    TransactionID      string
    TrackingNumber     string
    EstimatedDelivery  time.Time
    LoyaltyPointsEarned int
}

type CancellationResult struct {
    OrderID      string
    Status       string
    RefundAmount decimal.Decimal
    RefundMethod string
    ProcessedAt  time.Time
}

type OrderStatus struct {
    OrderID           string
    Status            string
    TrackingNumber    string
    EstimatedDelivery time.Time
    LastUpdated       time.Time
}

type ReturnRequest struct {
    OrderID      string
    Items        []OrderItem
    Reason       string
    RefundAmount decimal.Decimal
}

type ReturnResult struct {
    ReturnID            string
    Status              string
    RefundAmount        decimal.Decimal
    ReturnShippingLabel string
    ProcessedAt         time.Time
}

// Supporting types
type Customer struct {
    ID           string
    Email        string
    LoyaltyLevel string
}

type PaymentResult struct {
    TransactionID string
    Status        string
    Amount        decimal.Decimal
}

type ShippingQuote struct {
    Cost              decimal.Decimal
    EstimatedDelivery time.Time
    Carrier           string
}

type DeliverySchedule struct {
    TrackingNumber    string
    Carrier           string
    EstimatedDelivery time.Time
}

// Mock implementations for demonstration
type MockInventoryService struct{}

func (m *MockInventoryService) CheckInventory(productID string, quantity int) (bool, error) {
    return true, nil // Assume all items are in stock
}

func (m *MockInventoryService) ReserveItems(items []OrderItem) (string, error) {
    return "RES-" + generateID(), nil
}

func (m *MockInventoryService) ReleaseReservation(reservationID string) error {
    return nil
}

type MockPaymentService struct{}

func (m *MockPaymentService) ProcessPayment(payment *PaymentDetails) (*PaymentResult, error) {
    return &PaymentResult{
        TransactionID: "TXN-" + generateID(),
        Status:        "SUCCESS",
        Amount:        payment.Amount,
    }, nil
}

func (m *MockPaymentService) RefundPayment(transactionID string, amount decimal.Decimal) error {
    return nil
}

type MockShippingService struct{}

func (m *MockShippingService) CalculateShipping(address *Address, items []OrderItem) (*ShippingQuote, error) {
    return &ShippingQuote{
        Cost:              decimal.NewFromFloat(9.99),
        EstimatedDelivery: time.Now().Add(3 * 24 * time.Hour),
        Carrier:           "StandardShipping",
    }, nil
}

func (m *MockShippingService) ScheduleDelivery(order *Order) (*DeliverySchedule, error) {
    return &DeliverySchedule{
        TrackingNumber:    "TRK-" + generateID(),
        Carrier:           "StandardShipping",
        EstimatedDelivery: time.Now().Add(3 * 24 * time.Hour),
    }, nil
}

type MockNotificationService struct{}

func (m *MockNotificationService) SendOrderConfirmation(email string, order *Order) error {
    fmt.Printf("Sending order confirmation to %s for order %s\n", email, order.ID)
    return nil
}

func (m *MockNotificationService) SendShippingNotification(email string, trackingNumber string) error {
    fmt.Printf("Sending shipping notification to %s for tracking %s\n", email, trackingNumber)
    return nil
}

type MockTaxService struct{}

func (m *MockTaxService) CalculateTax(order *Order) (decimal.Decimal, error) {
    // Simple 8% tax rate
    return order.Subtotal.Mul(decimal.NewFromFloat(0.08)), nil
}

type MockCustomerService struct{}

func (m *MockCustomerService) GetCustomer(customerID string) (*Customer, error) {
    return &Customer{
        ID:           customerID,
        Email:        "customer@example.com",
        LoyaltyLevel: "GOLD",
    }, nil
}

func (m *MockCustomerService) UpdateLoyaltyPoints(customerID string, points int) error {
    fmt.Printf("Adding %d loyalty points to customer %s\n", points, customerID)
    return nil
}

// Helper functions
func generateOrderID() string {
    return "ORD-" + generateID()
}

func generateReturnID() string {
    return "RET-" + generateID()
}

func generateID() string {
    return fmt.Sprintf("%d", time.Now().UnixNano())
}

// Example usage
func main() {
    fmt.Println("=== Facade Pattern Demo ===\n")

    // Create logger
    logger, _ := zap.NewDevelopment()
    defer logger.Sync()

    // Create subsystem services (using mocks for demo)
    inventory := &MockInventoryService{}
    payment := &MockPaymentService{}
    shipping := &MockShippingService{}
    notifications := &MockNotificationService{}
    tax := &MockTaxService{}
    customer := &MockCustomerService{}

    // Create the facade
    orderFacade := NewEcommerceOrderFacade(
        inventory,
        payment,
        shipping,
        notifications,
        tax,
        customer,
        logger,
    )

    // Create a sample order request
    orderRequest := &OrderRequest{
        CustomerID: "CUST-12345",
        Items: []OrderItem{
            {
                ProductID: "PROD-001",
                Name:      "Wireless Headphones",
                Quantity:  1,
                Price:     decimal.NewFromFloat(99.99),
            },
            {
                ProductID: "PROD-002",
                Name:      "Phone Case",
                Quantity:  2,
                Price:     decimal.NewFromFloat(19.99),
            },
        },
        ShippingAddress: &Address{
            Street:  "123 Main St",
            City:    "Anytown",
            State:   "CA",
            ZipCode: "12345",
            Country: "USA",
        },
        BillingAddress: &Address{
            Street:  "123 Main St",
            City:    "Anytown",
            State:   "CA",
            ZipCode: "12345",
            Country: "USA",
        },
        PaymentDetails: &PaymentDetails{
            CardNumber:  "4111111111111111",
            ExpiryMonth: 12,
            ExpiryYear:  2025,
            CVV:         "123",
        },
    }

    // Place the order using the simplified facade interface
    fmt.Println("Placing order...")
    orderResult, err := orderFacade.PlaceOrder(context.Background(), orderRequest)
    if err != nil {
        fmt.Printf("Order failed: %v\n", err)
        return
    }

    fmt.Printf("Order placed successfully!\n")
    fmt.Printf("Order ID: %s\n", orderResult.OrderID)
    fmt.Printf("Status: %s\n", orderResult.Status)
    fmt.Printf("Total: $%s\n", orderResult.Total)
    fmt.Printf("Transaction ID: %s\n", orderResult.TransactionID)
    fmt.Printf("Tracking Number: %s\n", orderResult.TrackingNumber)
    fmt.Printf("Estimated Delivery: %s\n", orderResult.EstimatedDelivery.Format("2006-01-02"))
    fmt.Printf("Loyalty Points Earned: %d\n", orderResult.LoyaltyPointsEarned)

    // Demonstrate other facade operations
    fmt.Println("\nChecking order status...")
    status, err := orderFacade.GetOrderStatus(orderResult.OrderID)
    if err != nil {
        fmt.Printf("Failed to get order status: %v\n", err)
    } else {
        fmt.Printf("Current Status: %s\n", status.Status)
        fmt.Printf("Tracking: %s\n", status.TrackingNumber)
        fmt.Printf("Estimated Delivery: %s\n", status.EstimatedDelivery.Format("2006-01-02"))
    }

    // Demonstrate order cancellation
    fmt.Println("\nCancelling order...")
    cancellation, err := orderFacade.CancelOrder(orderResult.OrderID)
    if err != nil {
        fmt.Printf("Failed to cancel order: %v\n", err)
    } else {
        fmt.Printf("Cancellation processed\n")
        fmt.Printf("Refund Amount: $%s\n", cancellation.RefundAmount)
        fmt.Printf("Refund Method: %s\n", cancellation.RefundMethod)
    }

    // Demonstrate return processing
    fmt.Println("\nProcessing return...")
    returnRequest := &ReturnRequest{
        OrderID: orderResult.OrderID,
        Items: []OrderItem{
            orderRequest.Items[0], // Return first item
        },
        Reason:       "Defective product",
        RefundAmount: decimal.NewFromFloat(99.99),
    }

    returnResult, err := orderFacade.ProcessReturn(returnRequest)
    if err != nil {
        fmt.Printf("Failed to process return: %v\n", err)
    } else {
        fmt.Printf("Return processed\n")
        fmt.Printf("Return ID: %s\n", returnResult.ReturnID)
        fmt.Printf("Status: %s\n", returnResult.Status)
        fmt.Printf("Refund Amount: $%s\n", returnResult.RefundAmount)
        fmt.Printf("Return Label: %s\n", returnResult.ReturnShippingLabel)
    }

    fmt.Println("\n=== Facade Pattern Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Simple Facade (Basic Interface)**

```go
type SimpleFacade interface {
    Operation() error
}

type BasicFacade struct {
    subsystemA SubsystemA
    subsystemB SubsystemB
}

func (f *BasicFacade) Operation() error {
    f.subsystemA.OperationA()
    f.subsystemB.OperationB()
    return nil
}
```

2. **Context-Aware Facade**

```go
type ContextualFacade struct {
    subsystems map[string]Subsystem
    config     *Config
}

func (f *ContextualFacade) Execute(ctx context.Context, operation string) error {
    subsystem := f.selectSubsystem(ctx, operation)
    return subsystem.Execute(ctx, operation)
}

func (f *ContextualFacade) selectSubsystem(ctx context.Context, operation string) Subsystem {
    // Choose subsystem based on context
    if f.config.UseAlternative {
        return f.subsystems["alternative"]
    }
    return f.subsystems["default"]
}
```

3. **Configurable Facade**

```go
type ConfigurableFacade struct {
    components map[string]Component
    config     FacadeConfig
}

type FacadeConfig struct {
    EnableLogging     bool
    EnableValidation  bool
    EnableCache       bool
    TimeoutSeconds    int
}

func (f *ConfigurableFacade) ProcessRequest(request *Request) (*Response, error) {
    if f.config.EnableValidation {
        if err := f.components["validator"].Validate(request); err != nil {
            return nil, err
        }
    }

    if f.config.EnableCache {
        if cached := f.components["cache"].Get(request.ID); cached != nil {
            return cached.(*Response), nil
        }
    }

    response, err := f.components["processor"].Process(request)

    if f.config.EnableLogging {
        f.components["logger"].Log(request, response, err)
    }

    return response, err
}
```

### Trade-offs

**Pros:**

- **Simplified Interface**: Easier for clients to use
- **Decoupling**: Clients don't depend on subsystem details
- **Flexibility**: Can change subsystem implementations without affecting clients
- **Centralized Logic**: Common operations centralized in one place
- **Reduced Learning Curve**: Clients need to learn fewer interfaces

**Cons:**

- **Limited Functionality**: May not expose all subsystem capabilities
- **Additional Layer**: Adds another layer of indirection
- **Potential Bottleneck**: All operations go through the facade
- **God Object Risk**: Facade might become too large and complex
- **Maintenance Overhead**: Changes in subsystems may require facade updates

**When to Choose Facade vs Alternatives:**

| Scenario                  | Pattern     | Reason                       |
| ------------------------- | ----------- | ---------------------------- |
| Complex subsystem         | Facade      | Simplify client interactions |
| Microservices integration | API Gateway | Centralized entry point      |
| Legacy system wrapping    | Adapter     | Interface compatibility      |
| Cross-cutting concerns    | Decorator   | Layered functionality        |
| Algorithm selection       | Strategy    | Different algorithms         |

## Integration Tips

### 1. Builder Pattern Integration

```go
type OrderFacadeBuilder struct {
    inventory     InventoryService
    payment       PaymentService
    shipping      ShippingService
    notifications NotificationService
    tax           TaxService
    customer      CustomerService
    logger        *zap.Logger
    config        *FacadeConfig
}

func NewOrderFacadeBuilder() *OrderFacadeBuilder {
    return &OrderFacadeBuilder{
        config: &FacadeConfig{},
    }
}

func (b *OrderFacadeBuilder) WithInventoryService(service InventoryService) *OrderFacadeBuilder {
    b.inventory = service
    return b
}

func (b *OrderFacadeBuilder) WithPaymentService(service PaymentService) *OrderFacadeBuilder {
    b.payment = service
    return b
}

func (b *OrderFacadeBuilder) WithLogger(logger *zap.Logger) *OrderFacadeBuilder {
    b.logger = logger
    return b
}

func (b *OrderFacadeBuilder) Build() (*EcommerceOrderFacade, error) {
    if b.inventory == nil {
        return nil, fmt.Errorf("inventory service is required")
    }

    return NewEcommerceOrderFacade(
        b.inventory,
        b.payment,
        b.shipping,
        b.notifications,
        b.tax,
        b.customer,
        b.logger,
    ), nil
}
```

### 2. Factory Pattern Integration

```go
type FacadeFactory interface {
    CreateOrderFacade(config *Config) OrderFacade
    CreatePaymentFacade(config *Config) PaymentFacade
}

type DefaultFacadeFactory struct {
    serviceRegistry ServiceRegistry
}

func (f *DefaultFacadeFactory) CreateOrderFacade(config *Config) OrderFacade {
    return NewEcommerceOrderFacade(
        f.serviceRegistry.GetInventoryService(),
        f.serviceRegistry.GetPaymentService(),
        f.serviceRegistry.GetShippingService(),
        f.serviceRegistry.GetNotificationService(),
        f.serviceRegistry.GetTaxService(),
        f.serviceRegistry.GetCustomerService(),
        f.serviceRegistry.GetLogger(),
    )
}
```

### 3. Dependency Injection

```go
type ServiceContainer interface {
    RegisterService(name string, service interface{})
    GetService(name string) interface{}
}

type FacadeWithDI struct {
    container ServiceContainer
}

func (f *FacadeWithDI) ProcessOrder(orderRequest *OrderRequest) (*OrderResult, error) {
    inventory := f.container.GetService("inventory").(InventoryService)
    payment := f.container.GetService("payment").(PaymentService)

    // Use services...
    return &OrderResult{}, nil
}
```

### 4. Configuration-Driven Facade

```yaml
# facade_config.yaml
facades:
  order_processing:
    services:
      - name: inventory
        type: redis_inventory
        config:
          host: localhost:6379
      - name: payment
        type: stripe_payment
        config:
          api_key: ${STRIPE_API_KEY}
    features:
      enable_fraud_detection: true
      enable_loyalty_points: true
      max_retry_attempts: 3
```

```go
func CreateFacadeFromConfig(configPath string) (*EcommerceOrderFacade, error) {
    config, err := LoadConfig(configPath)
    if err != nil {
        return nil, err
    }

    var services []interface{}
    for _, serviceConfig := range config.Services {
        service, err := CreateService(serviceConfig.Type, serviceConfig.Config)
        if err != nil {
            return nil, err
        }
        services = append(services, service)
    }

    return BuildFacade(services, config.Features)
}
```

## Common Interview Questions

### 1. **How does Facade pattern differ from Adapter pattern?**

**Answer:**
Both are structural patterns but serve different purposes:

**Facade:**

```go
// Simplifies a complex subsystem
type PaymentFacade struct {
    validator PaymentValidator
    gateway   PaymentGateway
    logger    Logger
    auditor   Auditor
}

// One simple method hides complex workflow
func (f *PaymentFacade) ProcessPayment(payment *Payment) error {
    // Orchestrates multiple subsystems
    f.validator.Validate(payment)
    result := f.gateway.Process(payment)
    f.logger.Log(result)
    f.auditor.Audit(payment, result)
    return nil
}
```

**Adapter:**

```go
// Makes incompatible interfaces compatible
type LegacyPaymentGateway struct {
    // Old interface: ProcessLegacyPayment(string, float64)
}

type PaymentGatewayAdapter struct {
    legacy *LegacyPaymentGateway
}

// Adapts new interface to work with legacy system
func (a *PaymentGatewayAdapter) ProcessPayment(payment *Payment) error {
    // Convert new format to legacy format
    return a.legacy.ProcessLegacyPayment(payment.ToString(), payment.Amount.Float64())
}
```

**Key Differences:**

| Aspect           | Facade                           | Adapter                                 |
| ---------------- | -------------------------------- | --------------------------------------- |
| **Purpose**      | Simplify complex subsystem       | Make incompatible interfaces compatible |
| **Scope**        | Multiple classes/subsystems      | Usually single class/interface          |
| **Interface**    | Creates new simplified interface | Implements existing target interface    |
| **Relationship** | Uses subsystem classes           | Wraps adaptee class                     |
| **Intent**       | Hide complexity                  | Enable interoperability                 |

### 2. **How do you handle error propagation in Facade pattern?**

**Answer:**
Error handling in facades requires careful consideration of how to present subsystem errors to clients:

**Error Aggregation Strategy:**

```go
type FacadeError struct {
    Operation string
    Errors    []error
    Context   map[string]interface{}
}

func (f *FacadeError) Error() string {
    return fmt.Sprintf("facade operation '%s' failed with %d errors", f.Operation, len(f.Errors))
}

func (f *PaymentFacade) ProcessPayment(payment *Payment) error {
    var errors []error

    // Collect all validation errors
    if err := f.validator.ValidateCard(payment.Card); err != nil {
        errors = append(errors, fmt.Errorf("card validation: %w", err))
    }

    if err := f.validator.ValidateAmount(payment.Amount); err != nil {
        errors = append(errors, fmt.Errorf("amount validation: %w", err))
    }

    if len(errors) > 0 {
        return &FacadeError{
            Operation: "ProcessPayment",
            Errors:    errors,
            Context:   map[string]interface{}{"payment_id": payment.ID},
        }
    }

    // Continue with processing...
    return nil
}
```

**Failure Recovery Strategy:**

```go
func (f *PaymentFacade) ProcessPayment(payment *Payment) error {
    // Try primary gateway
    err := f.primaryGateway.Process(payment)
    if err == nil {
        return nil
    }

    // Log primary failure
    f.logger.Warn("Primary gateway failed, trying fallback", zap.Error(err))

    // Try fallback gateway
    err = f.fallbackGateway.Process(payment)
    if err == nil {
        f.logger.Info("Payment processed via fallback gateway")
        return nil
    }

    // Both failed
    return fmt.Errorf("payment processing failed on both gateways: primary=%v, fallback=%v", err, err)
}
```

**Circuit Breaker Integration:**

```go
type CircuitBreakerFacade struct {
    breakers map[string]*CircuitBreaker
    services map[string]interface{}
}

func (f *CircuitBreakerFacade) ProcessPayment(payment *Payment) error {
    // Use circuit breaker for each service
    err := f.breakers["gateway"].Execute(func() error {
        return f.services["gateway"].(PaymentGateway).Process(payment)
    })

    if err != nil {
        // Check if it's a circuit breaker error
        if circuitBreakerErr, ok := err.(*CircuitBreakerError); ok {
            // Handle circuit breaker open state
            return f.handleCircuitBreakerOpen(payment, circuitBreakerErr)
        }
        return err
    }

    return nil
}
```

### 3. **How do you implement async operations in Facade pattern?**

**Answer:**
Async operations in facades can be implemented using various Go concurrency patterns:

**Future/Promise Pattern:**

```go
type PaymentFuture struct {
    result chan *PaymentResult
    err    chan error
}

func (f *PaymentFuture) Get() (*PaymentResult, error) {
    select {
    case result := <-f.result:
        return result, nil
    case err := <-f.err:
        return nil, err
    }
}

func (pf *PaymentFacade) ProcessPaymentAsync(payment *Payment) *PaymentFuture {
    future := &PaymentFuture{
        result: make(chan *PaymentResult, 1),
        err:    make(chan error, 1),
    }

    go func() {
        result, err := pf.ProcessPayment(payment)
        if err != nil {
            future.err <- err
        } else {
            future.result <- result
        }
    }()

    return future
}

// Usage
future := facade.ProcessPaymentAsync(payment)
result, err := future.Get() // Blocks until completion
```

**Context with Timeout:**

```go
func (pf *PaymentFacade) ProcessPaymentWithTimeout(ctx context.Context, payment *Payment) (*PaymentResult, error) {
    resultChan := make(chan *PaymentResult, 1)
    errChan := make(chan error, 1)

    go func() {
        result, err := pf.ProcessPayment(payment)
        if err != nil {
            errChan <- err
        } else {
            resultChan <- result
        }
    }()

    select {
    case result := <-resultChan:
        return result, nil
    case err := <-errChan:
        return nil, err
    case <-ctx.Done():
        return nil, ctx.Err()
    }
}

// Usage with timeout
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

result, err := facade.ProcessPaymentWithTimeout(ctx, payment)
```

**Worker Pool Pattern:**

```go
type AsyncPaymentFacade struct {
    *PaymentFacade
    workers   int
    workChan  chan *AsyncPaymentRequest
    quit      chan struct{}
    wg        sync.WaitGroup
}

type AsyncPaymentRequest struct {
    Payment  *Payment
    Response chan *AsyncPaymentResponse
}

type AsyncPaymentResponse struct {
    Result *PaymentResult
    Error  error
}

func (apf *AsyncPaymentFacade) Start() {
    for i := 0; i < apf.workers; i++ {
        apf.wg.Add(1)
        go apf.worker()
    }
}

func (apf *AsyncPaymentFacade) worker() {
    defer apf.wg.Done()

    for {
        select {
        case req := <-apf.workChan:
            result, err := apf.PaymentFacade.ProcessPayment(req.Payment)
            req.Response <- &AsyncPaymentResponse{
                Result: result,
                Error:  err,
            }
        case <-apf.quit:
            return
        }
    }
}

func (apf *AsyncPaymentFacade) ProcessPaymentAsync(payment *Payment) <-chan *AsyncPaymentResponse {
    responseChan := make(chan *AsyncPaymentResponse, 1)

    request := &AsyncPaymentRequest{
        Payment:  payment,
        Response: responseChan,
    }

    select {
    case apf.workChan <- request:
        return responseChan
    default:
        // Queue is full
        responseChan <- &AsyncPaymentResponse{
            Error: fmt.Errorf("payment queue is full"),
        }
        return responseChan
    }
}
```

### 4. **How do you test Facade pattern effectively?**

**Answer:**
Testing facades requires both unit testing of the facade logic and integration testing with subsystems:

**Mock-based Unit Testing:**

```go
type MockPaymentGateway struct {
    mock.Mock
}

func (m *MockPaymentGateway) ProcessPayment(payment *Payment) (*PaymentResult, error) {
    args := m.Called(payment)
    return args.Get(0).(*PaymentResult), args.Error(1)
}

func TestPaymentFacade_ProcessPayment_Success(t *testing.T) {
    // Setup mocks
    mockValidator := &MockPaymentValidator{}
    mockGateway := &MockPaymentGateway{}
    mockLogger := &MockLogger{}

    // Create facade with mocks
    facade := NewPaymentFacade(mockValidator, mockGateway, mockLogger)

    // Setup expectations
    payment := &Payment{ID: "test-payment"}
    expectedResult := &PaymentResult{TransactionID: "txn-123"}

    mockValidator.On("Validate", payment).Return(nil)
    mockGateway.On("ProcessPayment", payment).Return(expectedResult, nil)
    mockLogger.On("Log", mock.Anything).Return(nil)

    // Execute
    result, err := facade.ProcessPayment(payment)

    // Assert
    assert.NoError(t, err)
    assert.Equal(t, expectedResult, result)

    // Verify all mocks were called
    mockValidator.AssertExpectations(t)
    mockGateway.AssertExpectations(t)
    mockLogger.AssertExpectations(t)
}
```

**Integration Testing:**

```go
func TestPaymentFacade_Integration(t *testing.T) {
    // Use real implementations or test doubles
    validator := NewRealPaymentValidator()
    gateway := NewTestPaymentGateway() // Test implementation
    logger := NewTestLogger()

    facade := NewPaymentFacade(validator, gateway, logger)

    // Test with real data flow
    payment := &Payment{
        Amount: decimal.NewFromFloat(100.00),
        Card:   &CreditCard{Number: "4111111111111111"},
    }

    result, err := facade.ProcessPayment(payment)

    assert.NoError(t, err)
    assert.NotNil(t, result)
    assert.NotEmpty(t, result.TransactionID)

    // Verify side effects
    assert.True(t, logger.HasLoggedSuccess())
    assert.Equal(t, 1, gateway.GetTransactionCount())
}
```

**Error Scenario Testing:**

```go
func TestPaymentFacade_ProcessPayment_ValidationError(t *testing.T) {
    mockValidator := &MockPaymentValidator{}
    mockGateway := &MockPaymentGateway{}

    facade := NewPaymentFacade(mockValidator, mockGateway, nil)

    payment := &Payment{ID: "invalid-payment"}
    validationError := fmt.Errorf("invalid card number")

    // Setup validation to fail
    mockValidator.On("Validate", payment).Return(validationError)

    // Gateway should not be called when validation fails
    mockGateway.AssertNotCalled(t, "ProcessPayment")

    result, err := facade.ProcessPayment(payment)

    assert.Error(t, err)
    assert.Nil(t, result)
    assert.Contains(t, err.Error(), "validation")

    mockValidator.AssertExpectations(t)
}
```

### 5. **How do you version and evolve Facade interfaces?**

**Answer:**
Facade versioning requires careful planning to maintain backward compatibility:

**Interface Versioning:**

```go
// Version 1
type PaymentFacadeV1 interface {
    ProcessPayment(payment *Payment) (*PaymentResult, error)
}

// Version 2 - extends V1
type PaymentFacadeV2 interface {
    PaymentFacadeV1
    ProcessPaymentWithOptions(payment *Payment, options *PaymentOptions) (*PaymentResult, error)
    RefundPayment(transactionID string, amount decimal.Decimal) (*RefundResult, error)
}

// Implementation supports both versions
type PaymentFacade struct {
    // ... fields
}

func (f *PaymentFacade) ProcessPayment(payment *Payment) (*PaymentResult, error) {
    // V1 implementation
    return f.ProcessPaymentWithOptions(payment, &PaymentOptions{})
}

func (f *PaymentFacade) ProcessPaymentWithOptions(payment *Payment, options *PaymentOptions) (*PaymentResult, error) {
    // V2 implementation with options
    return &PaymentResult{}, nil
}
```

**Backward Compatibility:**

```go
type LegacyPaymentFacade struct {
    modernFacade PaymentFacadeV2
}

// Maintain old method signatures
func (l *LegacyPaymentFacade) ProcessPayment(paymentData string) string {
    // Convert legacy format to modern format
    payment, err := l.convertLegacyPayment(paymentData)
    if err != nil {
        return "ERROR: " + err.Error()
    }

    result, err := l.modernFacade.ProcessPayment(payment)
    if err != nil {
        return "ERROR: " + err.Error()
    }

    // Convert modern result to legacy format
    return l.convertLegacyResult(result)
}
```

**Feature Flags:**

```go
type FeatureFlags struct {
    EnableNewPaymentFlow bool
    EnableFraudDetection bool
    EnableRetries        bool
}

type ConfigurablePaymentFacade struct {
    features FeatureFlags
    // ... other fields
}

func (f *ConfigurablePaymentFacade) ProcessPayment(payment *Payment) (*PaymentResult, error) {
    if f.features.EnableNewPaymentFlow {
        return f.processPaymentV2(payment)
    }
    return f.processPaymentV1(payment)
}
```
