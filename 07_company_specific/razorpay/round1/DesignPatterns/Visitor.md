---
# Auto-generated front matter
Title: Visitor
LastUpdated: 2025-11-06T20:45:58.514942
Tags: []
Status: draft
---

# Visitor Pattern

## Pattern Name & Intent

**Visitor** is a behavioral design pattern that allows you to define operations on a group of related objects without modifying their classes. It separates algorithms from the object structure on which they operate.

**Key Intent:**

- Define operations on object structures without modifying the classes
- Add new operations without changing existing object structures
- Separate algorithms from the data structures they operate on
- Support double dispatch to select the correct operation based on both visitor and element types
- Enable operations that require knowledge of multiple object types
- Facilitate navigation and processing of complex object hierarchies

## When to Use

**Use Visitor when:**

1. **Multiple Operations**: Need to perform many different operations on objects in a complex structure
2. **Stable Structure**: Object structure is stable but operations change frequently
3. **Type-specific Logic**: Operations depend on the concrete class of objects
4. **Traversal Logic**: Need to traverse complex object structures (trees, composites)
5. **Data Export**: Converting objects to different formats (JSON, XML, reports)
6. **Validation**: Performing various validations across object hierarchies
7. **Analysis**: Computing metrics or statistics across object collections

**Don't use when:**

- Object structure changes frequently (adding new types breaks visitors)
- Simple operations that don't justify the complexity
- Only one or two operations needed
- Objects have simple, uniform operations

## Real-World Use Cases (Payments/Fintech)

### 1. Financial Transaction Analysis

```go
// Financial transaction elements
type TransactionElement interface {
    Accept(visitor TransactionVisitor) interface{}
    GetTransactionID() string
    GetAmount() decimal.Decimal
    GetTimestamp() time.Time
}

// Transaction visitor interface
type TransactionVisitor interface {
    VisitPayment(payment *PaymentTransaction) interface{}
    VisitRefund(refund *RefundTransaction) interface{}
    VisitChargeback(chargeback *ChargebackTransaction) interface{}
    VisitAdjustment(adjustment *AdjustmentTransaction) interface{}
    VisitTransfer(transfer *TransferTransaction) interface{}
}

// Concrete transaction types
type PaymentTransaction struct {
    TransactionID   string
    Amount          decimal.Decimal
    Currency        string
    PaymentMethod   string
    MerchantID      string
    CustomerID      string
    Timestamp       time.Time
    GatewayFee      decimal.Decimal
    ProcessingFee   decimal.Decimal
    Status          string
    RiskScore       float64
    Metadata        map[string]interface{}
}

func (pt *PaymentTransaction) Accept(visitor TransactionVisitor) interface{} {
    return visitor.VisitPayment(pt)
}

func (pt *PaymentTransaction) GetTransactionID() string {
    return pt.TransactionID
}

func (pt *PaymentTransaction) GetAmount() decimal.Decimal {
    return pt.Amount
}

func (pt *PaymentTransaction) GetTimestamp() time.Time {
    return pt.Timestamp
}

type RefundTransaction struct {
    TransactionID     string
    OriginalPaymentID string
    Amount            decimal.Decimal
    Currency          string
    Reason            string
    RefundType        string // "FULL", "PARTIAL"
    Timestamp         time.Time
    ProcessingFee     decimal.Decimal
    Status            string
    Metadata          map[string]interface{}
}

func (rt *RefundTransaction) Accept(visitor TransactionVisitor) interface{} {
    return visitor.VisitRefund(rt)
}

func (rt *RefundTransaction) GetTransactionID() string {
    return rt.TransactionID
}

func (rt *RefundTransaction) GetAmount() decimal.Decimal {
    return rt.Amount
}

func (rt *RefundTransaction) GetTimestamp() time.Time {
    return rt.Timestamp
}

type ChargebackTransaction struct {
    TransactionID     string
    OriginalPaymentID string
    Amount            decimal.Decimal
    Currency          string
    Reason            string
    ReasonCode        string
    Timestamp         time.Time
    DisputeID         string
    Status            string
    Evidence          []string
    Metadata          map[string]interface{}
}

func (ct *ChargebackTransaction) Accept(visitor TransactionVisitor) interface{} {
    return visitor.VisitChargeback(ct)
}

func (ct *ChargebackTransaction) GetTransactionID() string {
    return ct.TransactionID
}

func (ct *ChargebackTransaction) GetAmount() decimal.Decimal {
    return ct.Amount
}

func (ct *ChargebackTransaction) GetTimestamp() time.Time {
    return ct.Timestamp
}

type AdjustmentTransaction struct {
    TransactionID   string
    Amount          decimal.Decimal
    Currency        string
    AdjustmentType  string // "FEE_CORRECTION", "BALANCE_ADJUSTMENT", "COMPENSATION"
    Reason          string
    Timestamp       time.Time
    ApprovedBy      string
    RelatedTransID  string
    Status          string
    Metadata        map[string]interface{}
}

func (at *AdjustmentTransaction) Accept(visitor TransactionVisitor) interface{} {
    return visitor.VisitAdjustment(at)
}

func (at *AdjustmentTransaction) GetTransactionID() string {
    return at.TransactionID
}

func (at *AdjustmentTransaction) GetAmount() decimal.Decimal {
    return at.Amount
}

func (at *AdjustmentTransaction) GetTimestamp() time.Time {
    return at.Timestamp
}

type TransferTransaction struct {
    TransactionID    string
    Amount           decimal.Decimal
    Currency         string
    FromAccountID    string
    ToAccountID      string
    TransferType     string // "INTERNAL", "EXTERNAL", "WIRE"
    Timestamp        time.Time
    ExchangeRate     decimal.Decimal
    TransferFee      decimal.Decimal
    Status           string
    SettlementDate   time.Time
    Metadata         map[string]interface{}
}

func (tt *TransferTransaction) Accept(visitor TransactionVisitor) interface{} {
    return visitor.VisitTransfer(tt)
}

func (tt *TransferTransaction) GetTransactionID() string {
    return tt.TransactionID
}

func (tt *TransferTransaction) GetAmount() decimal.Decimal {
    return tt.Amount
}

func (tt *TransferTransaction) GetTimestamp() time.Time {
    return tt.Timestamp
}

// Concrete visitor: Revenue Calculator
type RevenueCalculatorVisitor struct {
    totalRevenue    decimal.Decimal
    totalFees       decimal.Decimal
    totalRefunds    decimal.Decimal
    totalChargebacks decimal.Decimal
    transactionCount int
    logger          *zap.Logger
}

func NewRevenueCalculatorVisitor(logger *zap.Logger) *RevenueCalculatorVisitor {
    return &RevenueCalculatorVisitor{
        totalRevenue:     decimal.Zero,
        totalFees:        decimal.Zero,
        totalRefunds:     decimal.Zero,
        totalChargebacks: decimal.Zero,
        transactionCount: 0,
        logger:          logger,
    }
}

func (rcv *RevenueCalculatorVisitor) VisitPayment(payment *PaymentTransaction) interface{} {
    rcv.logger.Debug("Processing payment for revenue calculation",
        zap.String("transaction_id", payment.TransactionID),
        zap.String("amount", payment.Amount.String()))

    // Add payment amount to revenue
    rcv.totalRevenue = rcv.totalRevenue.Add(payment.Amount)

    // Add fees to total fees
    rcv.totalFees = rcv.totalFees.Add(payment.GatewayFee).Add(payment.ProcessingFee)

    rcv.transactionCount++

    return &RevenueImpact{
        Type:   "PAYMENT",
        Amount: payment.Amount,
        Fees:   payment.GatewayFee.Add(payment.ProcessingFee),
    }
}

func (rcv *RevenueCalculatorVisitor) VisitRefund(refund *RefundTransaction) interface{} {
    rcv.logger.Debug("Processing refund for revenue calculation",
        zap.String("transaction_id", refund.TransactionID),
        zap.String("amount", refund.Amount.String()))

    // Subtract refund amount from revenue
    rcv.totalRevenue = rcv.totalRevenue.Sub(refund.Amount)
    rcv.totalRefunds = rcv.totalRefunds.Add(refund.Amount)

    // Add processing fees
    rcv.totalFees = rcv.totalFees.Add(refund.ProcessingFee)

    rcv.transactionCount++

    return &RevenueImpact{
        Type:   "REFUND",
        Amount: refund.Amount.Neg(), // Negative impact
        Fees:   refund.ProcessingFee,
    }
}

func (rcv *RevenueCalculatorVisitor) VisitChargeback(chargeback *ChargebackTransaction) interface{} {
    rcv.logger.Debug("Processing chargeback for revenue calculation",
        zap.String("transaction_id", chargeback.TransactionID),
        zap.String("amount", chargeback.Amount.String()))

    // Subtract chargeback amount from revenue
    rcv.totalRevenue = rcv.totalRevenue.Sub(chargeback.Amount)
    rcv.totalChargebacks = rcv.totalChargebacks.Add(chargeback.Amount)

    // Chargebacks often incur additional fees
    chargebackFee := decimal.NewFromFloat(25.00) // Example chargeback fee
    rcv.totalFees = rcv.totalFees.Add(chargebackFee)

    rcv.transactionCount++

    return &RevenueImpact{
        Type:   "CHARGEBACK",
        Amount: chargeback.Amount.Neg(), // Negative impact
        Fees:   chargebackFee,
    }
}

func (rcv *RevenueCalculatorVisitor) VisitAdjustment(adjustment *AdjustmentTransaction) interface{} {
    rcv.logger.Debug("Processing adjustment for revenue calculation",
        zap.String("transaction_id", adjustment.TransactionID),
        zap.String("amount", adjustment.Amount.String()),
        zap.String("type", adjustment.AdjustmentType))

    // Handle different types of adjustments
    switch adjustment.AdjustmentType {
    case "FEE_CORRECTION":
        // Adjust fees
        rcv.totalFees = rcv.totalFees.Add(adjustment.Amount)
    case "BALANCE_ADJUSTMENT":
        // Adjust revenue
        rcv.totalRevenue = rcv.totalRevenue.Add(adjustment.Amount)
    case "COMPENSATION":
        // Subtract from revenue (cost to business)
        rcv.totalRevenue = rcv.totalRevenue.Sub(adjustment.Amount)
    }

    rcv.transactionCount++

    return &RevenueImpact{
        Type:   "ADJUSTMENT",
        Amount: adjustment.Amount,
        Fees:   decimal.Zero,
    }
}

func (rcv *RevenueCalculatorVisitor) VisitTransfer(transfer *TransferTransaction) interface{} {
    rcv.logger.Debug("Processing transfer for revenue calculation",
        zap.String("transaction_id", transfer.TransactionID),
        zap.String("amount", transfer.Amount.String()))

    // Transfers generate fee revenue
    rcv.totalFees = rcv.totalFees.Add(transfer.TransferFee)

    rcv.transactionCount++

    return &RevenueImpact{
        Type:   "TRANSFER",
        Amount: decimal.Zero, // Transfers don't directly impact revenue balance
        Fees:   transfer.TransferFee,
    }
}

func (rcv *RevenueCalculatorVisitor) GetTotalRevenue() decimal.Decimal {
    return rcv.totalRevenue
}

func (rcv *RevenueCalculatorVisitor) GetTotalFees() decimal.Decimal {
    return rcv.totalFees
}

func (rcv *RevenueCalculatorVisitor) GetNetRevenue() decimal.Decimal {
    return rcv.totalRevenue.Add(rcv.totalFees)
}

func (rcv *RevenueCalculatorVisitor) GetSummary() *RevenueSummary {
    return &RevenueSummary{
        TotalRevenue:     rcv.totalRevenue,
        TotalFees:        rcv.totalFees,
        NetRevenue:       rcv.GetNetRevenue(),
        TotalRefunds:     rcv.totalRefunds,
        TotalChargebacks: rcv.totalChargebacks,
        TransactionCount: rcv.transactionCount,
    }
}

// Concrete visitor: Risk Analyzer
type RiskAnalyzerVisitor struct {
    highRiskTransactions   []TransactionElement
    riskScoreSum          float64
    riskTransactionCount  int
    chargebackCount       int
    refundCount           int
    fraudIndicators       map[string]int
    logger               *zap.Logger
}

func NewRiskAnalyzerVisitor(logger *zap.Logger) *RiskAnalyzerVisitor {
    return &RiskAnalyzerVisitor{
        highRiskTransactions: make([]TransactionElement, 0),
        fraudIndicators:      make(map[string]int),
        logger:              logger,
    }
}

func (rav *RiskAnalyzerVisitor) VisitPayment(payment *PaymentTransaction) interface{} {
    rav.logger.Debug("Analyzing payment for risk",
        zap.String("transaction_id", payment.TransactionID),
        zap.Float64("risk_score", payment.RiskScore))

    rav.riskScoreSum += payment.RiskScore
    rav.riskTransactionCount++

    // High-risk payment detection
    if payment.RiskScore > 0.8 {
        rav.highRiskTransactions = append(rav.highRiskTransactions, payment)
        rav.fraudIndicators["HIGH_RISK_SCORE"]++
    }

    // Large amount detection
    if payment.Amount.GreaterThan(decimal.NewFromInt(10000)) {
        rav.fraudIndicators["LARGE_AMOUNT"]++
    }

    // Unusual payment method patterns
    if payment.PaymentMethod == "CRYPTOCURRENCY" {
        rav.fraudIndicators["CRYPTO_PAYMENT"]++
    }

    return &RiskAnalysis{
        TransactionID: payment.TransactionID,
        RiskScore:     payment.RiskScore,
        RiskLevel:     rav.calculateRiskLevel(payment.RiskScore),
        Indicators:    rav.getPaymentRiskIndicators(payment),
    }
}

func (rav *RiskAnalyzerVisitor) VisitRefund(refund *RefundTransaction) interface{} {
    rav.logger.Debug("Analyzing refund for risk",
        zap.String("transaction_id", refund.TransactionID))

    rav.refundCount++

    // Frequent refunds can indicate fraud
    if refund.Reason == "FRAUDULENT" {
        rav.fraudIndicators["FRAUD_REFUND"]++
    }

    // Large refund amounts
    if refund.Amount.GreaterThan(decimal.NewFromInt(5000)) {
        rav.fraudIndicators["LARGE_REFUND"]++
    }

    return &RiskAnalysis{
        TransactionID: refund.TransactionID,
        RiskScore:     0.3, // Default refund risk
        RiskLevel:     "MEDIUM",
        Indicators:    []string{"REFUND_TRANSACTION"},
    }
}

func (rav *RiskAnalyzerVisitor) VisitChargeback(chargeback *ChargebackTransaction) interface{} {
    rav.logger.Debug("Analyzing chargeback for risk",
        zap.String("transaction_id", chargeback.TransactionID),
        zap.String("reason_code", chargeback.ReasonCode))

    rav.chargebackCount++

    // Chargebacks are always high-risk indicators
    rav.highRiskTransactions = append(rav.highRiskTransactions, chargeback)
    rav.fraudIndicators["CHARGEBACK"]++

    // Fraud-related chargebacks
    if chargeback.ReasonCode == "4837" || chargeback.ReasonCode == "4863" { // Fraud codes
        rav.fraudIndicators["FRAUD_CHARGEBACK"]++
    }

    return &RiskAnalysis{
        TransactionID: chargeback.TransactionID,
        RiskScore:     1.0, // Chargebacks are always high risk
        RiskLevel:     "HIGH",
        Indicators:    []string{"CHARGEBACK", "DISPUTE"},
    }
}

func (rav *RiskAnalyzerVisitor) VisitAdjustment(adjustment *AdjustmentTransaction) interface{} {
    rav.logger.Debug("Analyzing adjustment for risk",
        zap.String("transaction_id", adjustment.TransactionID),
        zap.String("type", adjustment.AdjustmentType))

    // Manual adjustments can indicate issues
    if adjustment.AdjustmentType == "COMPENSATION" {
        rav.fraudIndicators["COMPENSATION"]++
    }

    return &RiskAnalysis{
        TransactionID: adjustment.TransactionID,
        RiskScore:     0.2, // Low risk for adjustments
        RiskLevel:     "LOW",
        Indicators:    []string{"ADJUSTMENT"},
    }
}

func (rav *RiskAnalyzerVisitor) VisitTransfer(transfer *TransferTransaction) interface{} {
    rav.logger.Debug("Analyzing transfer for risk",
        zap.String("transaction_id", transfer.TransactionID),
        zap.String("type", transfer.TransferType))

    // Large transfers can be risky
    if transfer.Amount.GreaterThan(decimal.NewFromInt(50000)) {
        rav.fraudIndicators["LARGE_TRANSFER"]++
    }

    // International transfers
    if transfer.TransferType == "WIRE" {
        rav.fraudIndicators["WIRE_TRANSFER"]++
    }

    return &RiskAnalysis{
        TransactionID: transfer.TransactionID,
        RiskScore:     0.4, // Medium risk for transfers
        RiskLevel:     "MEDIUM",
        Indicators:    rav.getTransferRiskIndicators(transfer),
    }
}

func (rav *RiskAnalyzerVisitor) calculateRiskLevel(score float64) string {
    if score >= 0.8 {
        return "HIGH"
    } else if score >= 0.5 {
        return "MEDIUM"
    }
    return "LOW"
}

func (rav *RiskAnalyzerVisitor) getPaymentRiskIndicators(payment *PaymentTransaction) []string {
    indicators := make([]string, 0)

    if payment.RiskScore > 0.8 {
        indicators = append(indicators, "HIGH_RISK_SCORE")
    }

    if payment.Amount.GreaterThan(decimal.NewFromInt(10000)) {
        indicators = append(indicators, "LARGE_AMOUNT")
    }

    if payment.PaymentMethod == "CRYPTOCURRENCY" {
        indicators = append(indicators, "CRYPTO_PAYMENT")
    }

    return indicators
}

func (rav *RiskAnalyzerVisitor) getTransferRiskIndicators(transfer *TransferTransaction) []string {
    indicators := make([]string, 0)

    if transfer.Amount.GreaterThan(decimal.NewFromInt(50000)) {
        indicators = append(indicators, "LARGE_TRANSFER")
    }

    if transfer.TransferType == "WIRE" {
        indicators = append(indicators, "WIRE_TRANSFER")
    }

    return indicators
}

func (rav *RiskAnalyzerVisitor) GetRiskSummary() *RiskSummary {
    averageRiskScore := 0.0
    if rav.riskTransactionCount > 0 {
        averageRiskScore = rav.riskScoreSum / float64(rav.riskTransactionCount)
    }

    return &RiskSummary{
        AverageRiskScore:     averageRiskScore,
        HighRiskCount:        len(rav.highRiskTransactions),
        ChargebackCount:      rav.chargebackCount,
        RefundCount:          rav.refundCount,
        FraudIndicators:      rav.fraudIndicators,
        TotalTransactions:    rav.riskTransactionCount,
    }
}

// Concrete visitor: Compliance Reporter
type ComplianceReporterVisitor struct {
    reportData      map[string]interface{}
    transactionLogs []ComplianceLog
    suspiciousCount int
    logger         *zap.Logger
}

func NewComplianceReporterVisitor(logger *zap.Logger) *ComplianceReporterVisitor {
    return &ComplianceReporterVisitor{
        reportData:      make(map[string]interface{}),
        transactionLogs: make([]ComplianceLog, 0),
        logger:         logger,
    }
}

func (crv *ComplianceReporterVisitor) VisitPayment(payment *PaymentTransaction) interface{} {
    crv.logger.Debug("Creating compliance log for payment",
        zap.String("transaction_id", payment.TransactionID))

    log := ComplianceLog{
        TransactionID:   payment.TransactionID,
        TransactionType: "PAYMENT",
        Amount:          payment.Amount,
        Currency:        payment.Currency,
        Timestamp:       payment.Timestamp,
        CustomerID:      payment.CustomerID,
        MerchantID:      payment.MerchantID,
        Status:          payment.Status,
        RiskScore:       payment.RiskScore,
    }

    // Check for suspicious activity
    if payment.Amount.GreaterThan(decimal.NewFromInt(10000)) {
        log.Flags = append(log.Flags, "LARGE_TRANSACTION")
        crv.suspiciousCount++
    }

    if payment.RiskScore > 0.9 {
        log.Flags = append(log.Flags, "HIGH_RISK")
        crv.suspiciousCount++
    }

    crv.transactionLogs = append(crv.transactionLogs, log)

    return &ComplianceRecord{
        TransactionID: payment.TransactionID,
        Logged:        true,
        Flags:         log.Flags,
    }
}

func (crv *ComplianceReporterVisitor) VisitRefund(refund *RefundTransaction) interface{} {
    crv.logger.Debug("Creating compliance log for refund",
        zap.String("transaction_id", refund.TransactionID))

    log := ComplianceLog{
        TransactionID:   refund.TransactionID,
        TransactionType: "REFUND",
        Amount:          refund.Amount,
        Currency:        refund.Currency,
        Timestamp:       refund.Timestamp,
        Status:          refund.Status,
        RelatedID:       refund.OriginalPaymentID,
    }

    // Refunds for fraud are reportable
    if refund.Reason == "FRAUDULENT" {
        log.Flags = append(log.Flags, "FRAUD_REFUND")
        crv.suspiciousCount++
    }

    crv.transactionLogs = append(crv.transactionLogs, log)

    return &ComplianceRecord{
        TransactionID: refund.TransactionID,
        Logged:        true,
        Flags:         log.Flags,
    }
}

func (crv *ComplianceReporterVisitor) VisitChargeback(chargeback *ChargebackTransaction) interface{} {
    crv.logger.Debug("Creating compliance log for chargeback",
        zap.String("transaction_id", chargeback.TransactionID))

    log := ComplianceLog{
        TransactionID:   chargeback.TransactionID,
        TransactionType: "CHARGEBACK",
        Amount:          chargeback.Amount,
        Currency:        chargeback.Currency,
        Timestamp:       chargeback.Timestamp,
        Status:          chargeback.Status,
        RelatedID:       chargeback.OriginalPaymentID,
    }

    // All chargebacks are flagged
    log.Flags = append(log.Flags, "CHARGEBACK")
    crv.suspiciousCount++

    // Fraud chargebacks get additional flags
    if chargeback.ReasonCode == "4837" || chargeback.ReasonCode == "4863" {
        log.Flags = append(log.Flags, "FRAUD_CHARGEBACK")
    }

    crv.transactionLogs = append(crv.transactionLogs, log)

    return &ComplianceRecord{
        TransactionID: chargeback.TransactionID,
        Logged:        true,
        Flags:         log.Flags,
    }
}

func (crv *ComplianceReporterVisitor) VisitAdjustment(adjustment *AdjustmentTransaction) interface{} {
    crv.logger.Debug("Creating compliance log for adjustment",
        zap.String("transaction_id", adjustment.TransactionID))

    log := ComplianceLog{
        TransactionID:   adjustment.TransactionID,
        TransactionType: "ADJUSTMENT",
        Amount:          adjustment.Amount,
        Currency:        adjustment.Currency,
        Timestamp:       adjustment.Timestamp,
        Status:          adjustment.Status,
        ApprovedBy:      adjustment.ApprovedBy,
    }

    // Manual adjustments require tracking
    log.Flags = append(log.Flags, "MANUAL_ADJUSTMENT")

    crv.transactionLogs = append(crv.transactionLogs, log)

    return &ComplianceRecord{
        TransactionID: adjustment.TransactionID,
        Logged:        true,
        Flags:         log.Flags,
    }
}

func (crv *ComplianceReporterVisitor) VisitTransfer(transfer *TransferTransaction) interface{} {
    crv.logger.Debug("Creating compliance log for transfer",
        zap.String("transaction_id", transfer.TransactionID))

    log := ComplianceLog{
        TransactionID:   transfer.TransactionID,
        TransactionType: "TRANSFER",
        Amount:          transfer.Amount,
        Currency:        transfer.Currency,
        Timestamp:       transfer.Timestamp,
        Status:          transfer.Status,
        FromAccount:     transfer.FromAccountID,
        ToAccount:       transfer.ToAccountID,
    }

    // Large transfers require reporting
    if transfer.Amount.GreaterThan(decimal.NewFromInt(10000)) {
        log.Flags = append(log.Flags, "LARGE_TRANSFER")
        crv.suspiciousCount++
    }

    // Wire transfers have additional requirements
    if transfer.TransferType == "WIRE" {
        log.Flags = append(log.Flags, "WIRE_TRANSFER")
    }

    crv.transactionLogs = append(crv.transactionLogs, log)

    return &ComplianceRecord{
        TransactionID: transfer.TransactionID,
        Logged:        true,
        Flags:         log.Flags,
    }
}

func (crv *ComplianceReporterVisitor) GenerateReport() *ComplianceReport {
    return &ComplianceReport{
        GeneratedAt:       time.Now(),
        TotalTransactions: len(crv.transactionLogs),
        SuspiciousCount:   crv.suspiciousCount,
        TransactionLogs:   crv.transactionLogs,
        Summary: map[string]interface{}{
            "total_logged":     len(crv.transactionLogs),
            "suspicious_count": crv.suspiciousCount,
            "compliance_rate":  float64(len(crv.transactionLogs)-crv.suspiciousCount) / float64(len(crv.transactionLogs)),
        },
    }
}

// Supporting types
type RevenueImpact struct {
    Type   string
    Amount decimal.Decimal
    Fees   decimal.Decimal
}

type RevenueSummary struct {
    TotalRevenue     decimal.Decimal
    TotalFees        decimal.Decimal
    NetRevenue       decimal.Decimal
    TotalRefunds     decimal.Decimal
    TotalChargebacks decimal.Decimal
    TransactionCount int
}

type RiskAnalysis struct {
    TransactionID string
    RiskScore     float64
    RiskLevel     string
    Indicators    []string
}

type RiskSummary struct {
    AverageRiskScore  float64
    HighRiskCount     int
    ChargebackCount   int
    RefundCount       int
    FraudIndicators   map[string]int
    TotalTransactions int
}

type ComplianceLog struct {
    TransactionID   string
    TransactionType string
    Amount          decimal.Decimal
    Currency        string
    Timestamp       time.Time
    CustomerID      string
    MerchantID      string
    Status          string
    RiskScore       float64
    RelatedID       string
    ApprovedBy      string
    FromAccount     string
    ToAccount       string
    Flags           []string
}

type ComplianceRecord struct {
    TransactionID string
    Logged        bool
    Flags         []string
}

type ComplianceReport struct {
    GeneratedAt       time.Time
    TotalTransactions int
    SuspiciousCount   int
    TransactionLogs   []ComplianceLog
    Summary           map[string]interface{}
}

// Transaction processor with visitor pattern
type TransactionProcessor struct {
    transactions []TransactionElement
    logger       *zap.Logger
}

func NewTransactionProcessor(logger *zap.Logger) *TransactionProcessor {
    return &TransactionProcessor{
        transactions: make([]TransactionElement, 0),
        logger:       logger,
    }
}

func (tp *TransactionProcessor) AddTransaction(transaction TransactionElement) {
    tp.transactions = append(tp.transactions, transaction)
}

func (tp *TransactionProcessor) ProcessWithVisitor(visitor TransactionVisitor) []interface{} {
    results := make([]interface{}, 0, len(tp.transactions))

    tp.logger.Info("Processing transactions with visitor",
        zap.Int("transaction_count", len(tp.transactions)),
        zap.String("visitor_type", fmt.Sprintf("%T", visitor)))

    for _, transaction := range tp.transactions {
        result := transaction.Accept(visitor)
        results = append(results, result)
    }

    return results
}

func (tp *TransactionProcessor) GetTransactionCount() int {
    return len(tp.transactions)
}

func (tp *TransactionProcessor) GetTransactions() []TransactionElement {
    return tp.transactions
}
```

### 2. Account Hierarchy Processing

```go
// Account hierarchy elements
type AccountElement interface {
    Accept(visitor AccountVisitor) interface{}
    GetAccountID() string
    GetAccountType() string
    GetBalance() decimal.Decimal
}

// Account visitor interface
type AccountVisitor interface {
    VisitSavingsAccount(account *SavingsAccount) interface{}
    VisitCheckingAccount(account *CheckingAccount) interface{}
    VisitCreditAccount(account *CreditAccount) interface{}
    VisitLoanAccount(account *LoanAccount) interface{}
    VisitInvestmentAccount(account *InvestmentAccount) interface{}
}

// Concrete account types
type SavingsAccount struct {
    AccountID     string
    CustomerID    string
    Balance       decimal.Decimal
    InterestRate  decimal.Decimal
    MinimumBalance decimal.Decimal
    Currency      string
    Status        string
    OpenedAt      time.Time
    LastActivity  time.Time
}

func (sa *SavingsAccount) Accept(visitor AccountVisitor) interface{} {
    return visitor.VisitSavingsAccount(sa)
}

func (sa *SavingsAccount) GetAccountID() string {
    return sa.AccountID
}

func (sa *SavingsAccount) GetAccountType() string {
    return "SAVINGS"
}

func (sa *SavingsAccount) GetBalance() decimal.Decimal {
    return sa.Balance
}

type CheckingAccount struct {
    AccountID      string
    CustomerID     string
    Balance        decimal.Decimal
    OverdraftLimit decimal.Decimal
    MonthlyFee     decimal.Decimal
    Currency       string
    Status         string
    OpenedAt       time.Time
    LastActivity   time.Time
}

func (ca *CheckingAccount) Accept(visitor AccountVisitor) interface{} {
    return visitor.VisitCheckingAccount(ca)
}

func (ca *CheckingAccount) GetAccountID() string {
    return ca.AccountID
}

func (ca *CheckingAccount) GetAccountType() string {
    return "CHECKING"
}

func (ca *CheckingAccount) GetBalance() decimal.Decimal {
    return ca.Balance
}

type CreditAccount struct {
    AccountID      string
    CustomerID     string
    Balance        decimal.Decimal // Outstanding amount owed
    CreditLimit    decimal.Decimal
    InterestRate   decimal.Decimal
    MinimumPayment decimal.Decimal
    DueDate        time.Time
    Currency       string
    Status         string
    OpenedAt       time.Time
    LastActivity   time.Time
}

func (ca *CreditAccount) Accept(visitor AccountVisitor) interface{} {
    return visitor.VisitCreditAccount(ca)
}

func (ca *CreditAccount) GetAccountID() string {
    return ca.AccountID
}

func (ca *CreditAccount) GetAccountType() string {
    return "CREDIT"
}

func (ca *CreditAccount) GetBalance() decimal.Decimal {
    return ca.Balance
}

// Interest calculation visitor
type InterestCalculatorVisitor struct {
    totalInterest   decimal.Decimal
    calculations    map[string]decimal.Decimal
    logger          *zap.Logger
}

func NewInterestCalculatorVisitor(logger *zap.Logger) *InterestCalculatorVisitor {
    return &InterestCalculatorVisitor{
        totalInterest: decimal.Zero,
        calculations:  make(map[string]decimal.Decimal),
        logger:       logger,
    }
}

func (icv *InterestCalculatorVisitor) VisitSavingsAccount(account *SavingsAccount) interface{} {
    // Calculate interest earned for savings account
    dailyRate := account.InterestRate.Div(decimal.NewFromInt(365))
    interest := account.Balance.Mul(dailyRate)

    icv.calculations[account.AccountID] = interest
    icv.totalInterest = icv.totalInterest.Add(interest)

    icv.logger.Debug("Calculated interest for savings account",
        zap.String("account_id", account.AccountID),
        zap.String("balance", account.Balance.String()),
        zap.String("interest", interest.String()))

    return &InterestCalculation{
        AccountID:    account.AccountID,
        AccountType:  "SAVINGS",
        Balance:      account.Balance,
        InterestRate: account.InterestRate,
        Interest:     interest,
        Type:         "EARNED",
    }
}

func (icv *InterestCalculatorVisitor) VisitCheckingAccount(account *CheckingAccount) interface{} {
    // Checking accounts typically don't earn interest
    interest := decimal.Zero

    icv.calculations[account.AccountID] = interest

    icv.logger.Debug("No interest for checking account",
        zap.String("account_id", account.AccountID))

    return &InterestCalculation{
        AccountID:   account.AccountID,
        AccountType: "CHECKING",
        Balance:     account.Balance,
        Interest:    interest,
        Type:        "NONE",
    }
}

func (icv *InterestCalculatorVisitor) VisitCreditAccount(account *CreditAccount) interface{} {
    // Calculate interest owed on credit account
    dailyRate := account.InterestRate.Div(decimal.NewFromInt(365))
    interest := account.Balance.Mul(dailyRate)

    icv.calculations[account.AccountID] = interest
    icv.totalInterest = icv.totalInterest.Sub(interest) // Interest owed is negative to total

    icv.logger.Debug("Calculated interest for credit account",
        zap.String("account_id", account.AccountID),
        zap.String("balance", account.Balance.String()),
        zap.String("interest", interest.String()))

    return &InterestCalculation{
        AccountID:    account.AccountID,
        AccountType:  "CREDIT",
        Balance:      account.Balance,
        InterestRate: account.InterestRate,
        Interest:     interest,
        Type:         "OWED",
    }
}

func (icv *InterestCalculatorVisitor) VisitLoanAccount(account *LoanAccount) interface{} {
    // Loans don't typically have daily interest calculations in this context
    interest := decimal.Zero

    icv.calculations[account.AccountID] = interest

    return &InterestCalculation{
        AccountID:   account.AccountID,
        AccountType: "LOAN",
        Balance:     account.Balance,
        Interest:    interest,
        Type:        "SCHEDULED",
    }
}

func (icv *InterestCalculatorVisitor) VisitInvestmentAccount(account *InvestmentAccount) interface{} {
    // Investment accounts might have dividend calculations
    interest := decimal.Zero // Simplified - would be based on portfolio performance

    icv.calculations[account.AccountID] = interest

    return &InterestCalculation{
        AccountID:   account.AccountID,
        AccountType: "INVESTMENT",
        Balance:     account.Balance,
        Interest:    interest,
        Type:        "VARIABLE",
    }
}

func (icv *InterestCalculatorVisitor) GetTotalInterest() decimal.Decimal {
    return icv.totalInterest
}

func (icv *InterestCalculatorVisitor) GetCalculations() map[string]decimal.Decimal {
    return icv.calculations
}

// Supporting types for account visitor
type InterestCalculation struct {
    AccountID    string
    AccountType  string
    Balance      decimal.Decimal
    InterestRate decimal.Decimal
    Interest     decimal.Decimal
    Type         string
}

type LoanAccount struct {
    AccountID        string
    CustomerID       string
    Balance          decimal.Decimal // Outstanding principal
    OriginalAmount   decimal.Decimal
    InterestRate     decimal.Decimal
    MonthlyPayment   decimal.Decimal
    RemainingTerms   int
    NextPaymentDate  time.Time
    Currency         string
    Status           string
    OpenedAt         time.Time
    LastActivity     time.Time
}

func (la *LoanAccount) Accept(visitor AccountVisitor) interface{} {
    return visitor.VisitLoanAccount(la)
}

func (la *LoanAccount) GetAccountID() string {
    return la.AccountID
}

func (la *LoanAccount) GetAccountType() string {
    return "LOAN"
}

func (la *LoanAccount) GetBalance() decimal.Decimal {
    return la.Balance
}

type InvestmentAccount struct {
    AccountID     string
    CustomerID    string
    Balance       decimal.Decimal // Total portfolio value
    CashBalance   decimal.Decimal
    Investments   map[string]decimal.Decimal
    Currency      string
    Status        string
    OpenedAt      time.Time
    LastActivity  time.Time
}

func (ia *InvestmentAccount) Accept(visitor AccountVisitor) interface{} {
    return visitor.VisitInvestmentAccount(ia)
}

func (ia *InvestmentAccount) GetAccountID() string {
    return ia.AccountID
}

func (ia *InvestmentAccount) GetAccountType() string {
    return "INVESTMENT"
}

func (ia *InvestmentAccount) GetBalance() decimal.Decimal {
    return ia.Balance
}
```

## Go Implementation

```go
package main

import (
    "fmt"
    "time"
    "go.uber.org/zap"
)

// Example: File system visitor for different operations
// Demonstrates visitor pattern with file system elements

// File system element interface
type FileSystemElement interface {
    Accept(visitor FileSystemVisitor) interface{}
    GetName() string
    GetSize() int64
    GetPath() string
}

// Visitor interface
type FileSystemVisitor interface {
    VisitFile(file *File) interface{}
    VisitDirectory(directory *Directory) interface{}
    VisitSymLink(symlink *SymLink) interface{}
}

// Concrete elements
type File struct {
    Name         string
    Path         string
    Size         int64
    Extension    string
    Content      []byte
    ModifiedTime time.Time
    Permissions  string
}

func (f *File) Accept(visitor FileSystemVisitor) interface{} {
    return visitor.VisitFile(f)
}

func (f *File) GetName() string {
    return f.Name
}

func (f *File) GetSize() int64 {
    return f.Size
}

func (f *File) GetPath() string {
    return f.Path
}

type Directory struct {
    Name         string
    Path         string
    Children     []FileSystemElement
    ModifiedTime time.Time
    Permissions  string
}

func (d *Directory) Accept(visitor FileSystemVisitor) interface{} {
    return visitor.VisitDirectory(d)
}

func (d *Directory) GetName() string {
    return d.Name
}

func (d *Directory) GetSize() int64 {
    // Directory size is sum of all children
    var totalSize int64
    for _, child := range d.Children {
        totalSize += child.GetSize()
    }
    return totalSize
}

func (d *Directory) GetPath() string {
    return d.Path
}

func (d *Directory) AddChild(element FileSystemElement) {
    d.Children = append(d.Children, element)
}

type SymLink struct {
    Name         string
    Path         string
    Target       string
    ModifiedTime time.Time
    Permissions  string
}

func (s *SymLink) Accept(visitor FileSystemVisitor) interface{} {
    return visitor.VisitSymLink(s)
}

func (s *SymLink) GetName() string {
    return s.Name
}

func (s *SymLink) GetSize() int64 {
    return 0 // Symlinks have no size
}

func (s *SymLink) GetPath() string {
    return s.Path
}

// Concrete visitor: Size calculator
type SizeCalculatorVisitor struct {
    totalSize      int64
    fileCount      int
    directoryCount int
    symlinkCount   int
    logger         *zap.Logger
}

func NewSizeCalculatorVisitor(logger *zap.Logger) *SizeCalculatorVisitor {
    return &SizeCalculatorVisitor{
        logger: logger,
    }
}

func (scv *SizeCalculatorVisitor) VisitFile(file *File) interface{} {
    scv.logger.Debug("Visiting file",
        zap.String("name", file.Name),
        zap.Int64("size", file.Size))

    scv.totalSize += file.Size
    scv.fileCount++

    return &SizeInfo{
        ElementType: "FILE",
        Name:        file.Name,
        Size:        file.Size,
        Count:       1,
    }
}

func (scv *SizeCalculatorVisitor) VisitDirectory(directory *Directory) interface{} {
    scv.logger.Debug("Visiting directory",
        zap.String("name", directory.Name),
        zap.Int("children", len(directory.Children)))

    scv.directoryCount++

    // Visit all children
    childrenSize := int64(0)
    for _, child := range directory.Children {
        result := child.Accept(scv)
        if sizeInfo, ok := result.(*SizeInfo); ok {
            childrenSize += sizeInfo.Size
        }
    }

    return &SizeInfo{
        ElementType: "DIRECTORY",
        Name:        directory.Name,
        Size:        childrenSize,
        Count:       len(directory.Children),
    }
}

func (scv *SizeCalculatorVisitor) VisitSymLink(symlink *SymLink) interface{} {
    scv.logger.Debug("Visiting symlink",
        zap.String("name", symlink.Name),
        zap.String("target", symlink.Target))

    scv.symlinkCount++

    return &SizeInfo{
        ElementType: "SYMLINK",
        Name:        symlink.Name,
        Size:        0,
        Count:       1,
    }
}

func (scv *SizeCalculatorVisitor) GetSummary() *SizeSummary {
    return &SizeSummary{
        TotalSize:      scv.totalSize,
        FileCount:      scv.fileCount,
        DirectoryCount: scv.directoryCount,
        SymlinkCount:   scv.symlinkCount,
        TotalElements:  scv.fileCount + scv.directoryCount + scv.symlinkCount,
    }
}

// Concrete visitor: Permission auditor
type PermissionAuditorVisitor struct {
    issues         []PermissionIssue
    checkedCount   int
    secureCount    int
    insecureCount  int
    logger         *zap.Logger
}

func NewPermissionAuditorVisitor(logger *zap.Logger) *PermissionAuditorVisitor {
    return &PermissionAuditorVisitor{
        issues: make([]PermissionIssue, 0),
        logger: logger,
    }
}

func (pav *PermissionAuditorVisitor) VisitFile(file *File) interface{} {
    pav.logger.Debug("Auditing file permissions",
        zap.String("name", file.Name),
        zap.String("permissions", file.Permissions))

    pav.checkedCount++

    issues := pav.checkFilePermissions(file)
    if len(issues) > 0 {
        pav.insecureCount++
        for _, issue := range issues {
            pav.issues = append(pav.issues, issue)
        }
    } else {
        pav.secureCount++
    }

    return &PermissionAuditResult{
        ElementType: "FILE",
        Name:        file.Name,
        Path:        file.Path,
        Permissions: file.Permissions,
        Issues:      issues,
        IsSecure:    len(issues) == 0,
    }
}

func (pav *PermissionAuditorVisitor) VisitDirectory(directory *Directory) interface{} {
    pav.logger.Debug("Auditing directory permissions",
        zap.String("name", directory.Name),
        zap.String("permissions", directory.Permissions))

    pav.checkedCount++

    issues := pav.checkDirectoryPermissions(directory)
    if len(issues) > 0 {
        pav.insecureCount++
        for _, issue := range issues {
            pav.issues = append(pav.issues, issue)
        }
    } else {
        pav.secureCount++
    }

    // Audit all children
    for _, child := range directory.Children {
        child.Accept(pav)
    }

    return &PermissionAuditResult{
        ElementType: "DIRECTORY",
        Name:        directory.Name,
        Path:        directory.Path,
        Permissions: directory.Permissions,
        Issues:      issues,
        IsSecure:    len(issues) == 0,
    }
}

func (pav *PermissionAuditorVisitor) VisitSymLink(symlink *SymLink) interface{} {
    pav.logger.Debug("Auditing symlink permissions",
        zap.String("name", symlink.Name),
        zap.String("permissions", symlink.Permissions))

    pav.checkedCount++

    issues := pav.checkSymlinkPermissions(symlink)
    if len(issues) > 0 {
        pav.insecureCount++
        for _, issue := range issues {
            pav.issues = append(pav.issues, issue)
        }
    } else {
        pav.secureCount++
    }

    return &PermissionAuditResult{
        ElementType: "SYMLINK",
        Name:        symlink.Name,
        Path:        symlink.Path,
        Permissions: symlink.Permissions,
        Issues:      issues,
        IsSecure:    len(issues) == 0,
    }
}

func (pav *PermissionAuditorVisitor) checkFilePermissions(file *File) []PermissionIssue {
    issues := make([]PermissionIssue, 0)

    // Check for world-writable files
    if pav.isWorldWritable(file.Permissions) {
        issues = append(issues, PermissionIssue{
            Severity:    "HIGH",
            Type:        "WORLD_WRITABLE",
            Description: "File is writable by everyone",
            Element:     file.Name,
            Path:        file.Path,
        })
    }

    // Check for executable files without proper permissions
    if file.Extension == ".exe" || file.Extension == ".sh" {
        if !pav.isExecutable(file.Permissions) {
            issues = append(issues, PermissionIssue{
                Severity:    "MEDIUM",
                Type:        "EXECUTABLE_NO_EXEC",
                Description: "Executable file without execute permission",
                Element:     file.Name,
                Path:        file.Path,
            })
        }
    }

    return issues
}

func (pav *PermissionAuditorVisitor) checkDirectoryPermissions(directory *Directory) []PermissionIssue {
    issues := make([]PermissionIssue, 0)

    // Check for world-writable directories
    if pav.isWorldWritable(directory.Permissions) {
        issues = append(issues, PermissionIssue{
            Severity:    "HIGH",
            Type:        "WORLD_WRITABLE_DIR",
            Description: "Directory is writable by everyone",
            Element:     directory.Name,
            Path:        directory.Path,
        })
    }

    // Check for directories without execute permission
    if !pav.isExecutable(directory.Permissions) {
        issues = append(issues, PermissionIssue{
            Severity:    "MEDIUM",
            Type:        "DIR_NO_EXEC",
            Description: "Directory without execute permission",
            Element:     directory.Name,
            Path:        directory.Path,
        })
    }

    return issues
}

func (pav *PermissionAuditorVisitor) checkSymlinkPermissions(symlink *SymLink) []PermissionIssue {
    issues := make([]PermissionIssue, 0)

    // Check for potentially dangerous symlinks
    if symlink.Target == "/" || symlink.Target == "/etc" {
        issues = append(issues, PermissionIssue{
            Severity:    "HIGH",
            Type:        "DANGEROUS_SYMLINK",
            Description: "Symlink points to sensitive system directory",
            Element:     symlink.Name,
            Path:        symlink.Path,
        })
    }

    return issues
}

func (pav *PermissionAuditorVisitor) isWorldWritable(permissions string) bool {
    // Simplified check - in real implementation would parse permission string
    return len(permissions) >= 3 && permissions[2] == 'w'
}

func (pav *PermissionAuditorVisitor) isExecutable(permissions string) bool {
    // Simplified check - in real implementation would parse permission string
    return len(permissions) >= 1 && permissions[0] == 'x'
}

func (pav *PermissionAuditorVisitor) GetAuditSummary() *PermissionAuditSummary {
    return &PermissionAuditSummary{
        TotalChecked:  pav.checkedCount,
        SecureCount:   pav.secureCount,
        InsecureCount: pav.insecureCount,
        Issues:        pav.issues,
        SecurityScore: float64(pav.secureCount) / float64(pav.checkedCount) * 100,
    }
}

// Concrete visitor: Backup selector
type BackupSelectorVisitor struct {
    selectedFiles    []FileSystemElement
    totalSize        int64
    selectionCriteria BackupCriteria
    logger           *zap.Logger
}

type BackupCriteria struct {
    MaxFileSize      int64
    IncludeExtensions []string
    ExcludeExtensions []string
    ModifiedSince    time.Time
    IncludeHidden    bool
}

func NewBackupSelectorVisitor(criteria BackupCriteria, logger *zap.Logger) *BackupSelectorVisitor {
    return &BackupSelectorVisitor{
        selectedFiles:     make([]FileSystemElement, 0),
        selectionCriteria: criteria,
        logger:           logger,
    }
}

func (bsv *BackupSelectorVisitor) VisitFile(file *File) interface{} {
    bsv.logger.Debug("Evaluating file for backup",
        zap.String("name", file.Name),
        zap.Int64("size", file.Size))

    shouldBackup := bsv.shouldBackupFile(file)

    if shouldBackup {
        bsv.selectedFiles = append(bsv.selectedFiles, file)
        bsv.totalSize += file.Size
    }

    return &BackupSelection{
        ElementType: "FILE",
        Name:        file.Name,
        Path:        file.Path,
        Size:        file.Size,
        Selected:    shouldBackup,
        Reason:      bsv.getSelectionReason(file, shouldBackup),
    }
}

func (bsv *BackupSelectorVisitor) VisitDirectory(directory *Directory) interface{} {
    bsv.logger.Debug("Evaluating directory for backup",
        zap.String("name", directory.Name))

    // Visit all children
    selectedChildren := 0
    for _, child := range directory.Children {
        result := child.Accept(bsv)
        if selection, ok := result.(*BackupSelection); ok && selection.Selected {
            selectedChildren++
        }
    }

    return &BackupSelection{
        ElementType: "DIRECTORY",
        Name:        directory.Name,
        Path:        directory.Path,
        Size:        directory.GetSize(),
        Selected:    selectedChildren > 0,
        Reason:      fmt.Sprintf("Contains %d selected files", selectedChildren),
    }
}

func (bsv *BackupSelectorVisitor) VisitSymLink(symlink *SymLink) interface{} {
    bsv.logger.Debug("Evaluating symlink for backup",
        zap.String("name", symlink.Name))

    // Usually don't backup symlinks
    return &BackupSelection{
        ElementType: "SYMLINK",
        Name:        symlink.Name,
        Path:        symlink.Path,
        Size:        0,
        Selected:    false,
        Reason:      "Symlinks not included in backup",
    }
}

func (bsv *BackupSelectorVisitor) shouldBackupFile(file *File) bool {
    // Check file size limit
    if bsv.selectionCriteria.MaxFileSize > 0 && file.Size > bsv.selectionCriteria.MaxFileSize {
        return false
    }

    // Check modification time
    if !bsv.selectionCriteria.ModifiedSince.IsZero() &&
       file.ModifiedTime.Before(bsv.selectionCriteria.ModifiedSince) {
        return false
    }

    // Check hidden files
    if !bsv.selectionCriteria.IncludeHidden && file.Name[0] == '.' {
        return false
    }

    // Check excluded extensions
    for _, ext := range bsv.selectionCriteria.ExcludeExtensions {
        if file.Extension == ext {
            return false
        }
    }

    // Check included extensions (if specified)
    if len(bsv.selectionCriteria.IncludeExtensions) > 0 {
        for _, ext := range bsv.selectionCriteria.IncludeExtensions {
            if file.Extension == ext {
                return true
            }
        }
        return false
    }

    return true
}

func (bsv *BackupSelectorVisitor) getSelectionReason(file *File, selected bool) string {
    if !selected {
        if file.Size > bsv.selectionCriteria.MaxFileSize {
            return "File too large"
        }
        if !bsv.selectionCriteria.ModifiedSince.IsZero() &&
           file.ModifiedTime.Before(bsv.selectionCriteria.ModifiedSince) {
            return "Not modified recently"
        }
        if !bsv.selectionCriteria.IncludeHidden && file.Name[0] == '.' {
            return "Hidden file excluded"
        }
        for _, ext := range bsv.selectionCriteria.ExcludeExtensions {
            if file.Extension == ext {
                return "Extension excluded"
            }
        }
        return "Extension not included"
    }
    return "Meets backup criteria"
}

func (bsv *BackupSelectorVisitor) GetBackupSummary() *BackupSummary {
    return &BackupSummary{
        SelectedCount: len(bsv.selectedFiles),
        TotalSize:     bsv.totalSize,
        SelectedFiles: bsv.selectedFiles,
        Criteria:      bsv.selectionCriteria,
    }
}

// Supporting types
type SizeInfo struct {
    ElementType string
    Name        string
    Size        int64
    Count       int
}

type SizeSummary struct {
    TotalSize      int64
    FileCount      int
    DirectoryCount int
    SymlinkCount   int
    TotalElements  int
}

type PermissionIssue struct {
    Severity    string
    Type        string
    Description string
    Element     string
    Path        string
}

type PermissionAuditResult struct {
    ElementType string
    Name        string
    Path        string
    Permissions string
    Issues      []PermissionIssue
    IsSecure    bool
}

type PermissionAuditSummary struct {
    TotalChecked  int
    SecureCount   int
    InsecureCount int
    Issues        []PermissionIssue
    SecurityScore float64
}

type BackupSelection struct {
    ElementType string
    Name        string
    Path        string
    Size        int64
    Selected    bool
    Reason      string
}

type BackupSummary struct {
    SelectedCount int
    TotalSize     int64
    SelectedFiles []FileSystemElement
    Criteria      BackupCriteria
}

// Example usage
func main() {
    fmt.Println("=== Visitor Pattern Demo ===\n")

    // Create logger
    logger, _ := zap.NewDevelopment()
    defer logger.Sync()

    // Create file system structure
    root := &Directory{
        Name:         "root",
        Path:         "/",
        ModifiedTime: time.Now(),
        Permissions:  "rwxr-xr-x",
    }

    // Add files to root
    root.AddChild(&File{
        Name:         "document.txt",
        Path:         "/document.txt",
        Size:         1024,
        Extension:    ".txt",
        ModifiedTime: time.Now().AddDate(0, 0, -1),
        Permissions:  "rw-r--r--",
    })

    root.AddChild(&File{
        Name:         "script.sh",
        Path:         "/script.sh",
        Size:         512,
        Extension:    ".sh",
        ModifiedTime: time.Now(),
        Permissions:  "rwxr-xr-x",
    })

    root.AddChild(&File{
        Name:         "large_file.dat",
        Path:         "/large_file.dat",
        Size:         10485760, // 10MB
        Extension:    ".dat",
        ModifiedTime: time.Now().AddDate(0, 0, -7),
        Permissions:  "rw-r--r--",
    })

    // Add subdirectory
    subdir := &Directory{
        Name:         "subdir",
        Path:         "/subdir",
        ModifiedTime: time.Now(),
        Permissions:  "rwxr-xr-x",
    }

    subdir.AddChild(&File{
        Name:         "config.json",
        Path:         "/subdir/config.json",
        Size:         256,
        Extension:    ".json",
        ModifiedTime: time.Now(),
        Permissions:  "rw-r--r--",
    })

    subdir.AddChild(&SymLink{
        Name:         "link_to_etc",
        Path:         "/subdir/link_to_etc",
        Target:       "/etc",
        ModifiedTime: time.Now(),
        Permissions:  "rwxrwxrwx",
    })

    root.AddChild(subdir)

    // Example 1: Size Calculator Visitor
    fmt.Println("=== Size Calculator ===")
    sizeCalculator := NewSizeCalculatorVisitor(logger)
    root.Accept(sizeCalculator)

    sizeSummary := sizeCalculator.GetSummary()
    fmt.Printf("Total Size: %d bytes\n", sizeSummary.TotalSize)
    fmt.Printf("Files: %d\n", sizeSummary.FileCount)
    fmt.Printf("Directories: %d\n", sizeSummary.DirectoryCount)
    fmt.Printf("Symlinks: %d\n", sizeSummary.SymlinkCount)
    fmt.Printf("Total Elements: %d\n", sizeSummary.TotalElements)
    fmt.Println()

    // Example 2: Permission Auditor Visitor
    fmt.Println("=== Permission Auditor ===")
    permissionAuditor := NewPermissionAuditorVisitor(logger)
    root.Accept(permissionAuditor)

    auditSummary := permissionAuditor.GetAuditSummary()
    fmt.Printf("Total Checked: %d\n", auditSummary.TotalChecked)
    fmt.Printf("Secure: %d\n", auditSummary.SecureCount)
    fmt.Printf("Insecure: %d\n", auditSummary.InsecureCount)
    fmt.Printf("Security Score: %.1f%%\n", auditSummary.SecurityScore)

    if len(auditSummary.Issues) > 0 {
        fmt.Println("Security Issues:")
        for _, issue := range auditSummary.Issues {
            fmt.Printf("  [%s] %s: %s (%s)\n",
                issue.Severity, issue.Element, issue.Description, issue.Type)
        }
    }
    fmt.Println()

    // Example 3: Backup Selector Visitor
    fmt.Println("=== Backup Selector ===")
    criteria := BackupCriteria{
        MaxFileSize:       5242880, // 5MB
        IncludeExtensions: []string{".txt", ".json", ".sh"},
        ExcludeExtensions: []string{".tmp", ".log"},
        ModifiedSince:     time.Now().AddDate(0, 0, -3), // Last 3 days
        IncludeHidden:     false,
    }

    backupSelector := NewBackupSelectorVisitor(criteria, logger)
    root.Accept(backupSelector)

    backupSummary := backupSelector.GetBackupSummary()
    fmt.Printf("Selected Files: %d\n", backupSummary.SelectedCount)
    fmt.Printf("Total Backup Size: %d bytes\n", backupSummary.TotalSize)

    fmt.Println("Selected for backup:")
    for _, file := range backupSummary.SelectedFiles {
        fmt.Printf("  %s (%d bytes)\n", file.GetName(), file.GetSize())
    }
    fmt.Println()

    // Example 4: Demonstrate Visitor Pattern Benefits
    fmt.Println("=== Visitor Pattern Benefits ===")
    fmt.Println("1. Added three different operations without modifying file system classes")
    fmt.Println("2. Each visitor encapsulates a specific algorithm")
    fmt.Println("3. Easy to add new operations by creating new visitors")
    fmt.Println("4. Type-safe double dispatch ensures correct method is called")
    fmt.Println("5. Separation of concerns - algorithms separate from data structures")
    fmt.Println()

    fmt.Println("=== Visitor Pattern Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Acyclic Visitor**

```go
// Avoids cyclic dependencies between visitor and elements
type VisitorBase interface{}

type FileVisitor interface {
    VisitorBase
    VisitFile(file *File)
}

type DirectoryVisitor interface {
    VisitorBase
    VisitDirectory(directory *Directory)
}

type Element interface {
    Accept(visitor VisitorBase)
}

func (f *File) Accept(visitor VisitorBase) {
    if fileVisitor, ok := visitor.(FileVisitor); ok {
        fileVisitor.VisitFile(f)
    }
}
```

2. **Reflective Visitor**

```go
// Uses reflection to dispatch to correct method
type ReflectiveVisitor struct {
    methods map[reflect.Type]reflect.Value
}

func (rv *ReflectiveVisitor) Visit(element interface{}) interface{} {
    elementType := reflect.TypeOf(element)
    if method, exists := rv.methods[elementType]; exists {
        results := method.Call([]reflect.Value{reflect.ValueOf(element)})
        if len(results) > 0 {
            return results[0].Interface()
        }
    }
    return nil
}
```

3. **Hierarchical Visitor**

```go
// Supports inheritance hierarchies with fallback methods
type HierarchicalVisitor interface {
    VisitDefault(element interface{}) interface{}
    GetSpecificMethod(elementType reflect.Type) (reflect.Value, bool)
}

func (hv *BaseHierarchicalVisitor) Visit(element interface{}) interface{} {
    elementType := reflect.TypeOf(element)

    if method, exists := hv.GetSpecificMethod(elementType); exists {
        return method.Call([]reflect.Value{reflect.ValueOf(element)})[0].Interface()
    }

    return hv.VisitDefault(element)
}
```

### Trade-offs

**Pros:**

- **Separation of Concerns**: Operations separate from data structures
- **Extensibility**: Easy to add new operations without modifying existing classes
- **Type Safety**: Compile-time checking of visitor-element combinations
- **Single Responsibility**: Each visitor handles one specific operation
- **Polymorphism**: Correct method called based on both visitor and element types

**Cons:**

- **Complexity**: Adds significant complexity for simple operations
- **Brittle Structure**: Adding new element types requires updating all visitors
- **Circular Dependencies**: Visitors and elements often have circular dependencies
- **Performance**: Double dispatch can be slower than direct method calls
- **Hard to Understand**: Flow of control can be difficult to follow

## Integration Tips

### 1. Builder Pattern Integration

```go
// Use builder to construct complex visitor configurations
type VisitorBuilder struct {
    visitors []TransactionVisitor
    filters  []TransactionFilter
    config   VisitorConfig
}

func (vb *VisitorBuilder) AddVisitor(visitor TransactionVisitor) *VisitorBuilder {
    vb.visitors = append(vb.visitors, visitor)
    return vb
}

func (vb *VisitorBuilder) AddFilter(filter TransactionFilter) *VisitorBuilder {
    vb.filters = append(vb.filters, filter)
    return vb
}

func (vb *VisitorBuilder) Build() *CompositeVisitor {
    return &CompositeVisitor{
        visitors: vb.visitors,
        filters:  vb.filters,
        config:   vb.config,
    }
}
```

### 2. Strategy Pattern Integration

```go
// Combine visitor with strategy for flexible processing
type VisitorStrategy interface {
    CreateVisitor() TransactionVisitor
}

type RevenueAnalysisStrategy struct{}
func (ras *RevenueAnalysisStrategy) CreateVisitor() TransactionVisitor {
    return NewRevenueCalculatorVisitor()
}

type RiskAnalysisStrategy struct{}
func (ras *RiskAnalysisStrategy) CreateVisitor() TransactionVisitor {
    return NewRiskAnalyzerVisitor()
}

type VisitorProcessor struct {
    strategy VisitorStrategy
}

func (vp *VisitorProcessor) SetStrategy(strategy VisitorStrategy) {
    vp.strategy = strategy
}

func (vp *VisitorProcessor) Process(transactions []TransactionElement) {
    visitor := vp.strategy.CreateVisitor()
    for _, transaction := range transactions {
        transaction.Accept(visitor)
    }
}
```

### 3. Chain of Responsibility Integration

```go
// Chain multiple visitors together
type VisitorChain struct {
    visitors []TransactionVisitor
    next     *VisitorChain
}

func (vc *VisitorChain) Process(transaction TransactionElement) []interface{} {
    results := make([]interface{}, 0)

    for _, visitor := range vc.visitors {
        result := transaction.Accept(visitor)
        results = append(results, result)
    }

    if vc.next != nil {
        nextResults := vc.next.Process(transaction)
        results = append(results, nextResults...)
    }

    return results
}
```

## Common Interview Questions

### 1. **How does Visitor pattern achieve double dispatch?**

**Answer:**
Double dispatch means the method to execute depends on both the type of the visitor and the type of the element being visited.

```go
// First dispatch: based on element type
func (payment *PaymentTransaction) Accept(visitor TransactionVisitor) interface{} {
    // Second dispatch: based on visitor interface and element type
    return visitor.VisitPayment(payment)
}

// This ensures the correct combination is called
func (calculator *RevenueCalculatorVisitor) VisitPayment(payment *PaymentTransaction) interface{} {
    // Revenue calculation specific to payments
    return calculator.calculatePaymentRevenue(payment)
}

func (analyzer *RiskAnalyzerVisitor) VisitPayment(payment *PaymentTransaction) interface{} {
    // Risk analysis specific to payments
    return analyzer.analyzePaymentRisk(payment)
}
```

**Why double dispatch matters:**

1. **Type Safety**: Compiler ensures visitor has method for element type
2. **Polymorphism**: Correct method called without type checking
3. **Extensibility**: New visitors work with existing elements
4. **Performance**: No reflection or type switching needed

**Comparison with single dispatch:**

```go
// Single dispatch (visitor has to check types)
func (visitor *GenericVisitor) Process(element interface{}) {
    switch e := element.(type) {
    case *PaymentTransaction:
        visitor.processPayment(e)
    case *RefundTransaction:
        visitor.processRefund(e)
    default:
        visitor.processDefault(e)
    }
}

// Double dispatch (type system handles dispatch)
func (payment *PaymentTransaction) Accept(visitor TransactionVisitor) {
    return visitor.VisitPayment(payment) // Automatic dispatch
}
```

### 2. **When would you choose Visitor over other behavioral patterns?**

**Answer:**

**Choose Visitor when:**

- Object structure is stable but operations change frequently
- Need to perform multiple unrelated operations on object hierarchy
- Want to avoid polluting classes with unrelated methods
- Operations require knowledge of multiple element types

**Don't choose Visitor when:**

- Object structure changes frequently
- Only simple, uniform operations needed
- Performance is critical
- Operations are closely related to object behavior

**Comparison with alternatives:**

| Pattern      | Use Case                                | Pros                                    | Cons                                   |
| ------------ | --------------------------------------- | --------------------------------------- | -------------------------------------- |
| **Visitor**  | Multiple operations on stable structure | Clean separation, extensible operations | Complex, brittle to structure changes  |
| **Strategy** | Different algorithms for same operation | Runtime algorithm switching             | Single operation focus                 |
| **Command**  | Encapsulate operations as objects       | Undo/redo, queuing                      | Operation-centric, not structure-aware |
| **Observer** | Notify multiple objects of changes      | Loose coupling, dynamic subscription    | Event-driven, not operation-driven     |

**Example decision matrix:**

```go
type PatternDecision struct {
    StructureStability string // "stable", "changing"
    OperationCount     int
    OperationComplexity string // "simple", "complex"
    RuntimeFlexibility bool
    PerformanceNeeds   string // "low", "high"
}

func (pd *PatternDecision) RecommendPattern() string {
    if pd.StructureStability == "stable" && pd.OperationCount > 3 && pd.OperationComplexity == "complex" {
        return "Visitor"
    }

    if pd.RuntimeFlexibility && pd.OperationCount <= 3 {
        return "Strategy"
    }

    if pd.StructureStability == "changing" || pd.PerformanceNeeds == "high" {
        return "Direct methods"
    }

    return "Visitor"
}
```

### 3. **How do you handle the problem of adding new element types to Visitor pattern?**

**Answer:**
Adding new element types to Visitor pattern is its main weakness - it requires modifying all existing visitors.

**Problem illustration:**

```go
// Original visitor interface
type TransactionVisitor interface {
    VisitPayment(payment *PaymentTransaction) interface{}
    VisitRefund(refund *RefundTransaction) interface{}
}

// Adding new element requires changing ALL visitors
type TransactionVisitor interface {
    VisitPayment(payment *PaymentTransaction) interface{}
    VisitRefund(refund *RefundTransaction) interface{}
    VisitSubscription(sub *SubscriptionTransaction) interface{} // NEW!
}
```

**Solution strategies:**

**1. Default Implementation Pattern:**

```go
// Provide default implementations
type BaseTransactionVisitor struct{}

func (btv *BaseTransactionVisitor) VisitPayment(payment *PaymentTransaction) interface{} {
    return btv.VisitDefault(payment)
}

func (btv *BaseTransactionVisitor) VisitRefund(refund *RefundTransaction) interface{} {
    return btv.VisitDefault(refund)
}

func (btv *BaseTransactionVisitor) VisitSubscription(sub *SubscriptionTransaction) interface{} {
    return btv.VisitDefault(sub)
}

func (btv *BaseTransactionVisitor) VisitDefault(element TransactionElement) interface{} {
    // Default behavior for all elements
    return nil
}

// Concrete visitors only override what they need
type RevenueCalculatorVisitor struct {
    BaseTransactionVisitor
}

func (rcv *RevenueCalculatorVisitor) VisitPayment(payment *PaymentTransaction) interface{} {
    // Only implement payment handling
}
```

**2. Extensible Visitor Pattern:**

```go
// Use map-based dispatch
type ExtensibleVisitor struct {
    handlers map[reflect.Type]func(interface{}) interface{}
}

func (ev *ExtensibleVisitor) Register(elementType reflect.Type, handler func(interface{}) interface{}) {
    ev.handlers[elementType] = handler
}

func (ev *ExtensibleVisitor) Visit(element interface{}) interface{} {
    elementType := reflect.TypeOf(element)
    if handler, exists := ev.handlers[elementType]; exists {
        return handler(element)
    }
    return ev.defaultHandler(element)
}

// Usage
visitor := &ExtensibleVisitor{handlers: make(map[reflect.Type]func(interface{}) interface{})}
visitor.Register(reflect.TypeOf(&PaymentTransaction{}), func(e interface{}) interface{} {
    return processPayment(e.(*PaymentTransaction))
})
// Adding new type doesn't break existing code
visitor.Register(reflect.TypeOf(&SubscriptionTransaction{}), func(e interface{}) interface{} {
    return processSubscription(e.(*SubscriptionTransaction))
})
```

**3. Interface Segregation:**

```go
// Split visitor into focused interfaces
type PaymentVisitor interface {
    VisitPayment(payment *PaymentTransaction) interface{}
}

type RefundVisitor interface {
    VisitRefund(refund *RefundTransaction) interface{}
}

type SubscriptionVisitor interface {
    VisitSubscription(sub *SubscriptionTransaction) interface{}
}

// Elements accept specific visitor types
func (payment *PaymentTransaction) Accept(visitor PaymentVisitor) interface{} {
    return visitor.VisitPayment(payment)
}

// Visitors implement only interfaces they need
type RevenueCalculator struct{}

func (rc *RevenueCalculator) VisitPayment(payment *PaymentTransaction) interface{} {
    // Implementation
}

func (rc *RevenueCalculator) VisitRefund(refund *RefundTransaction) interface{} {
    // Implementation
}
// Don't need to implement SubscriptionVisitor if not needed
```

### 4. **How do you test Visitor pattern implementations?**

**Answer:**

**1. Mock Elements:**

```go
type MockTransaction struct {
    mockID     string
    mockAmount decimal.Decimal
    visitCount int
}

func (mt *MockTransaction) Accept(visitor TransactionVisitor) interface{} {
    mt.visitCount++
    if paymentVisitor, ok := visitor.(interface {
        VisitMockTransaction(*MockTransaction) interface{}
    }); ok {
        return paymentVisitor.VisitMockTransaction(mt)
    }
    return nil
}

func TestVisitorWithMockElements(t *testing.T) {
    visitor := NewRevenueCalculatorVisitor()
    mockTx := &MockTransaction{mockID: "test", mockAmount: decimal.NewFromInt(100)}

    result := mockTx.Accept(visitor)

    assert.Equal(t, 1, mockTx.visitCount)
    assert.NotNil(t, result)
}
```

**2. Test Each Visit Method:**

```go
func TestRevenueCalculatorVisitor(t *testing.T) {
    tests := []struct {
        name        string
        transaction TransactionElement
        expectedRevenue decimal.Decimal
    }{
        {
            name: "payment adds to revenue",
            transaction: &PaymentTransaction{
                Amount: decimal.NewFromInt(100),
            },
            expectedRevenue: decimal.NewFromInt(100),
        },
        {
            name: "refund subtracts from revenue",
            transaction: &RefundTransaction{
                Amount: decimal.NewFromInt(50),
            },
            expectedRevenue: decimal.NewFromInt(-50),
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            visitor := NewRevenueCalculatorVisitor()
            result := tt.transaction.Accept(visitor)

            if revenueImpact, ok := result.(*RevenueImpact); ok {
                assert.Equal(t, tt.expectedRevenue, revenueImpact.Amount)
            }
        })
    }
}
```

**3. Integration Testing:**

```go
func TestVisitorIntegration(t *testing.T) {
    processor := NewTransactionProcessor()

    // Add various transaction types
    processor.AddTransaction(&PaymentTransaction{Amount: decimal.NewFromInt(100)})
    processor.AddTransaction(&RefundTransaction{Amount: decimal.NewFromInt(30)})
    processor.AddTransaction(&ChargebackTransaction{Amount: decimal.NewFromInt(20)})

    // Test with revenue calculator
    revenueVisitor := NewRevenueCalculatorVisitor()
    results := processor.ProcessWithVisitor(revenueVisitor)

    assert.Equal(t, 3, len(results))

    summary := revenueVisitor.GetSummary()
    expectedNet := decimal.NewFromInt(100).Sub(decimal.NewFromInt(30)).Sub(decimal.NewFromInt(20))
    assert.Equal(t, expectedNet, summary.TotalRevenue)
}
```

**4. Visitor State Testing:**

```go
func TestVisitorStatefulBehavior(t *testing.T) {
    visitor := NewRiskAnalyzerVisitor()

    // Process high-risk payment
    highRiskPayment := &PaymentTransaction{
        RiskScore: 0.9,
        Amount:    decimal.NewFromInt(15000),
    }

    visitor.VisitPayment(highRiskPayment)

    // Verify state changes
    summary := visitor.GetRiskSummary()
    assert.Equal(t, 1, summary.HighRiskCount)
    assert.Contains(t, summary.FraudIndicators, "HIGH_RISK_SCORE")
    assert.Contains(t, summary.FraudIndicators, "LARGE_AMOUNT")
}
```

### 5. **How do you handle error scenarios in Visitor pattern?**

**Answer:**

**1. Error Return Strategy:**

```go
type VisitorResult struct {
    Data  interface{}
    Error error
}

type ErrorHandlingVisitor interface {
    VisitPayment(payment *PaymentTransaction) VisitorResult
    VisitRefund(refund *RefundTransaction) VisitorResult
}

func (calculator *RevenueCalculatorVisitor) VisitPayment(payment *PaymentTransaction) VisitorResult {
    if payment.Amount.LessThan(decimal.Zero) {
        return VisitorResult{
            Error: fmt.Errorf("invalid payment amount: %s", payment.Amount),
        }
    }

    revenue := calculator.calculateRevenue(payment)
    return VisitorResult{
        Data: revenue,
    }
}
```

**2. Error Collection Pattern:**

```go
type ErrorCollectingVisitor struct {
    errors   []error
    results  []interface{}
    continueOnError bool
}

func (ecv *ErrorCollectingVisitor) VisitPayment(payment *PaymentTransaction) interface{} {
    result, err := ecv.processPayment(payment)
    if err != nil {
        ecv.errors = append(ecv.errors, err)
        if !ecv.continueOnError {
            return nil
        }
    }

    ecv.results = append(ecv.results, result)
    return result
}

func (ecv *ErrorCollectingVisitor) GetErrors() []error {
    return ecv.errors
}

func (ecv *ErrorCollectingVisitor) HasErrors() bool {
    return len(ecv.errors) > 0
}
```

**3. Try-Catch Style Pattern:**

```go
type SafeVisitor struct {
    BaseVisitor
    errorHandler func(error, TransactionElement)
    logger       *zap.Logger
}

func (sv *SafeVisitor) VisitPayment(payment *PaymentTransaction) (result interface{}) {
    defer func() {
        if r := recover(); r != nil {
            err := fmt.Errorf("panic in payment visitor: %v", r)
            sv.logger.Error("Visitor panic", zap.Error(err))
            if sv.errorHandler != nil {
                sv.errorHandler(err, payment)
            }
            result = nil
        }
    }()

    return sv.BaseVisitor.VisitPayment(payment)
}
```

**4. Circuit Breaker Integration:**

```go
type CircuitBreakerVisitor struct {
    BaseVisitor
    breaker *CircuitBreaker
    logger  *zap.Logger
}

func (cbv *CircuitBreakerVisitor) VisitPayment(payment *PaymentTransaction) interface{} {
    return cbv.breaker.Execute(func() (interface{}, error) {
        result := cbv.BaseVisitor.VisitPayment(payment)
        if result == nil {
            return nil, fmt.Errorf("visitor returned nil result")
        }
        return result, nil
    })
}
```

**5. Validation Before Processing:**

```go
type ValidatingVisitor struct {
    BaseVisitor
    validator TransactionValidator
}

func (vv *ValidatingVisitor) VisitPayment(payment *PaymentTransaction) interface{} {
    if err := vv.validator.ValidatePayment(payment); err != nil {
        return &VisitorError{
            Type:    "VALIDATION_ERROR",
            Message: err.Error(),
            Element: payment,
        }
    }

    return vv.BaseVisitor.VisitPayment(payment)
}

type VisitorError struct {
    Type    string
    Message string
    Element TransactionElement
}

func (ve *VisitorError) Error() string {
    return fmt.Sprintf("%s: %s", ve.Type, ve.Message)
}
```
