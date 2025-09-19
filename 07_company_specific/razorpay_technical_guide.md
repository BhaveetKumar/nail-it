# 💳 **Razorpay Technical Deep Dive & Interview Guide**

*Complete preparation guide for backend engineering roles at Razorpay - covering fintech systems, payment gateways, compliance, and scalability challenges*

---

## 📋 **Table of Contents**

1. [Company Overview & Architecture](#-company-overview--architecture)
2. [Payment Gateway Systems](#-payment-gateway-systems)
3. [Financial Compliance & Security](#-financial-compliance--security)
4. [Real-Time Processing & Streaming](#-real-time-processing--streaming)
5. [Fraud Detection & Risk Management](#-fraud-detection--risk-management)
6. [Microservices Architecture](#-microservices-architecture)
7. [Database Design & Scaling](#-database-design--scaling)
8. [API Design & Rate Limiting](#-api-design--rate-limiting)
9. [Monitoring & Observability](#-monitoring--observability)
10. [Interview Questions & Scenarios](#-interview-questions--scenarios)

---

## 🏢 **Company Overview & Architecture**

### **Razorpay Business Model**

Razorpay is India's leading fintech company providing payment solutions including:

- **Payment Gateway**: Online payment processing
- **Payment Links**: Simple payment collection
- **Payment Pages**: Customizable checkout pages
- **Subscriptions**: Recurring payment management
- **Route**: Marketplace payment splitting
- **Capital**: Business lending
- **Payroll**: Salary and compliance management

### **Technical Architecture Overview**

```go
package razorpay

import (
    "context"
    "time"
)

// Core Razorpay Architecture Components
type RazorpayArchitecture struct {
    // Payment Processing Core
    PaymentGateway    *PaymentGatewayService
    PaymentProcessor  *PaymentProcessorService
    SettlementEngine  *SettlementService
    
    // Financial Services
    ComplianceEngine  *ComplianceService
    FraudDetection    *FraudDetectionService
    RiskManagement    *RiskManagementService
    
    // Business Services
    MerchantOnboarding *MerchantService
    SubscriptionEngine *SubscriptionService
    MarketplaceRoute   *RouteService
    
    // Infrastructure
    EventBus          *EventBusService
    NotificationService *NotificationService
    AuditService      *AuditService
    
    // External Integrations
    BankingPartners   []BankingPartner
    PaymentNetworks   []PaymentNetwork
    RegulatoryAPIs    []RegulatoryAPI
}

// Core payment processing pipeline
type PaymentPipeline struct {
    Validation    *ValidationStage
    Authentication *AuthenticationStage
    Authorization  *AuthorizationStage
    Processing     *ProcessingStage
    Settlement     *SettlementStage
    Reconciliation *ReconciliationStage
}

type PaymentRequest struct {
    MerchantID      string                 `json:"merchant_id"`
    Amount          int64                  `json:"amount"` // in paise
    Currency        string                 `json:"currency"`
    PaymentMethod   PaymentMethod          `json:"payment_method"`
    CustomerDetails CustomerDetails        `json:"customer"`
    OrderDetails    OrderDetails           `json:"order"`
    Metadata        map[string]interface{} `json:"metadata"`
    CallbackURL     string                 `json:"callback_url"`
    WebhookURL      string                 `json:"webhook_url"`
    Timestamp       time.Time              `json:"timestamp"`
}

type PaymentMethod struct {
    Type        PaymentType        `json:"type"`
    Card        *CardDetails       `json:"card,omitempty"`
    NetBanking  *NetBankingDetails `json:"netbanking,omitempty"`
    UPI         *UPIDetails        `json:"upi,omitempty"`
    Wallet      *WalletDetails     `json:"wallet,omitempty"`
    EMI         *EMIDetails        `json:"emi,omitempty"`
}

type PaymentType string

const (
    PaymentTypeCard       PaymentType = "card"
    PaymentTypeNetBanking PaymentType = "netbanking"
    PaymentTypeUPI        PaymentType = "upi"
    PaymentTypeWallet     PaymentType = "wallet"
    PaymentTypeEMI        PaymentType = "emi"
    PaymentTypeCOD        PaymentType = "cod"
    PaymentTypeBankTransfer PaymentType = "bank_transfer"
)

type PaymentStatus string

const (
    PaymentStatusCreated    PaymentStatus = "created"
    PaymentStatusAuthorized PaymentStatus = "authorized"
    PaymentStatusCaptured   PaymentStatus = "captured"
    PaymentStatusFailed     PaymentStatus = "failed"
    PaymentStatusRefunded   PaymentStatus = "refunded"
    PaymentStatusDisputed   PaymentStatus = "disputed"
)
```

### **Scaling Challenges at Razorpay**

**Volume Requirements:**
- 100+ million transactions/month
- Peak: 50,000+ TPS during festivals
- Sub-second payment processing
- 99.9% uptime requirement

**Geographical Distribution:**
- Multi-region deployment (Mumbai, Bangalore, Delhi)
- CDN for static assets
- Edge computing for latency reduction

---

## 💳 **Payment Gateway Systems**

### **Payment Processing Architecture**

```go
package gateway

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Payment Gateway Service - Core of Razorpay
type PaymentGatewayService struct {
    router           *PaymentRouter
    processors       map[PaymentType]PaymentProcessor
    fraudDetector    *FraudDetectionService
    complianceEngine *ComplianceEngine
    settlementEngine *SettlementEngine
    auditLogger      *AuditLogger
    metrics          *PaymentMetrics
    
    // Circuit breakers for external services
    bankCircuitBreakers map[string]*CircuitBreaker
    
    mu sync.RWMutex
}

type PaymentProcessor interface {
    ProcessPayment(ctx context.Context, req PaymentRequest) (*PaymentResponse, error)
    CapturePayment(ctx context.Context, paymentID string, amount int64) error
    RefundPayment(ctx context.Context, paymentID string, amount int64) error
    GetPaymentStatus(ctx context.Context, paymentID string) (PaymentStatus, error)
}

type PaymentRouter struct {
    routingRules    []RoutingRule
    fallbackRoutes  map[PaymentType][]string
    loadBalancer    *PaymentLoadBalancer
    healthChecker   *BankHealthChecker
}

type RoutingRule struct {
    Condition    RoutingCondition `json:"condition"`
    Destination  string          `json:"destination"`
    Priority     int             `json:"priority"`
    Weight       int             `json:"weight"`
    Active       bool            `json:"active"`
}

type RoutingCondition struct {
    PaymentMethod   *PaymentType `json:"payment_method,omitempty"`
    AmountRange     *AmountRange `json:"amount_range,omitempty"`
    MerchantTier    *string      `json:"merchant_tier,omitempty"`
    CustomerRegion  *string      `json:"customer_region,omitempty"`
    TimeOfDay       *TimeRange   `json:"time_of_day,omitempty"`
    BankCode        *string      `json:"bank_code,omitempty"`
    SuccessRate     *float64     `json:"success_rate,omitempty"`
}

func NewPaymentGatewayService(config GatewayConfig) *PaymentGatewayService {
    pgs := &PaymentGatewayService{
        router:              NewPaymentRouter(config.RoutingConfig),
        processors:          make(map[PaymentType]PaymentProcessor),
        fraudDetector:       NewFraudDetectionService(config.FraudConfig),
        complianceEngine:    NewComplianceEngine(config.ComplianceConfig),
        settlementEngine:    NewSettlementEngine(config.SettlementConfig),
        auditLogger:         NewAuditLogger(config.AuditConfig),
        metrics:             NewPaymentMetrics(),
        bankCircuitBreakers: make(map[string]*CircuitBreaker),
    }
    
    // Initialize payment processors
    pgs.initializeProcessors(config)
    
    // Setup circuit breakers for banking partners
    pgs.setupCircuitBreakers(config.BankingPartners)
    
    return pgs
}

func (pgs *PaymentGatewayService) ProcessPayment(ctx context.Context, req PaymentRequest) (*PaymentResponse, error) {
    startTime := time.Now()
    
    // Generate unique payment ID
    paymentID := generatePaymentID()
    req.PaymentID = paymentID
    
    // Audit log - request received
    pgs.auditLogger.LogPaymentRequest(paymentID, req)
    
    // Pre-processing validations
    if err := pgs.validatePaymentRequest(req); err != nil {
        pgs.metrics.RecordValidationFailure(req.PaymentMethod.Type)
        return nil, fmt.Errorf("validation failed: %w", err)
    }
    
    // Fraud detection
    fraudResult := pgs.fraudDetector.AnalyzePayment(ctx, req)
    if fraudResult.Action == FraudActionBlock {
        pgs.metrics.RecordFraudBlocked(req.PaymentMethod.Type)
        pgs.auditLogger.LogFraudBlocked(paymentID, fraudResult)
        return nil, ErrPaymentBlocked
    }
    
    // Compliance checks
    if err := pgs.complianceEngine.ValidatePayment(ctx, req); err != nil {
        pgs.metrics.RecordComplianceFailure(req.PaymentMethod.Type)
        return nil, fmt.Errorf("compliance validation failed: %w", err)
    }
    
    // Route payment to appropriate processor
    processor, err := pgs.router.RoutePayment(ctx, req)
    if err != nil {
        pgs.metrics.RecordRoutingFailure(req.PaymentMethod.Type)
        return nil, fmt.Errorf("payment routing failed: %w", err)
    }
    
    // Process payment with circuit breaker
    circuitBreaker := pgs.getCircuitBreaker(processor.GetBankCode())
    
    response, err := circuitBreaker.Execute(ctx, func() (interface{}, error) {
        return processor.ProcessPayment(ctx, req)
    })
    
    processingTime := time.Since(startTime)
    
    if err != nil {
        pgs.metrics.RecordPaymentFailure(req.PaymentMethod.Type, processingTime)
        pgs.auditLogger.LogPaymentFailure(paymentID, err)
        return nil, err
    }
    
    paymentResponse := response.(*PaymentResponse)
    
    // Post-processing
    if paymentResponse.Status == PaymentStatusAuthorized {
        // Schedule settlement
        pgs.settlementEngine.ScheduleSettlement(ctx, paymentResponse)
        
        // Send success webhook
        go pgs.sendWebhook(ctx, req.WebhookURL, paymentResponse)
    }
    
    pgs.metrics.RecordPaymentSuccess(req.PaymentMethod.Type, processingTime)
    pgs.auditLogger.LogPaymentSuccess(paymentID, paymentResponse)
    
    return paymentResponse, nil
}

// Card Payment Processor
type CardPaymentProcessor struct {
    acquirerGateways map[string]AcquirerGateway
    tokenVault       *TokenVault
    pciCompliance    *PCIComplianceService
    threeDSService   *ThreeDSecureService
    binRangeService  *BINRangeService
}

type AcquirerGateway interface {
    Authorize(ctx context.Context, req CardAuthRequest) (*CardAuthResponse, error)
    Capture(ctx context.Context, authCode string, amount int64) error
    Void(ctx context.Context, authCode string) error
    Refund(ctx context.Context, transactionID string, amount int64) error
}

func (cpp *CardPaymentProcessor) ProcessPayment(ctx context.Context, req PaymentRequest) (*PaymentResponse, error) {
    cardDetails := req.PaymentMethod.Card
    
    // PCI compliance check
    if err := cpp.pciCompliance.ValidateCardData(cardDetails); err != nil {
        return nil, fmt.Errorf("PCI validation failed: %w", err)
    }
    
    // BIN range validation
    binInfo, err := cpp.binRangeService.GetBINInfo(cardDetails.Number[:6])
    if err != nil {
        return nil, fmt.Errorf("BIN validation failed: %w", err)
    }
    
    // 3D Secure check
    if binInfo.Requires3DS {
        threeDSResult, err := cpp.threeDSService.Authenticate(ctx, cardDetails, req.Amount)
        if err != nil {
            return nil, fmt.Errorf("3DS authentication failed: %w", err)
        }
        
        if !threeDSResult.Authenticated {
            return &PaymentResponse{
                PaymentID: req.PaymentID,
                Status:    PaymentStatusFailed,
                Error:     "3DS authentication failed",
            }, nil
        }
    }
    
    // Tokenize card for security
    token, err := cpp.tokenVault.TokenizeCard(cardDetails)
    if err != nil {
        return nil, fmt.Errorf("card tokenization failed: %w", err)
    }
    
    // Select acquirer based on routing rules
    acquirer := cpp.selectAcquirer(binInfo, req.Amount)
    
    // Create authorization request
    authReq := CardAuthRequest{
        Token:        token,
        Amount:       req.Amount,
        Currency:     req.Currency,
        MerchantID:   req.MerchantID,
        OrderID:      req.OrderDetails.OrderID,
        CustomerInfo: req.CustomerDetails,
    }
    
    // Authorize with acquirer
    authResp, err := acquirer.Authorize(ctx, authReq)
    if err != nil {
        return nil, fmt.Errorf("authorization failed: %w", err)
    }
    
    status := PaymentStatusFailed
    if authResp.Approved {
        status = PaymentStatusAuthorized
    }
    
    return &PaymentResponse{
        PaymentID:       req.PaymentID,
        Status:          status,
        AuthCode:        authResp.AuthCode,
        TransactionID:   authResp.TransactionID,
        ProcessedAmount: req.Amount,
        Currency:        req.Currency,
        ProcessedAt:     time.Now(),
        AcquirerResponse: authResp,
    }, nil
}

// UPI Payment Processor
type UPIPaymentProcessor struct {
    upiGateways    map[string]UPIGateway
    vpaValidator   *VPAValidator
    upiIDService   *UPIIDService
    qrService      *QRCodeService
}

type UPIGateway interface {
    InitiatePayment(ctx context.Context, req UPIPaymentRequest) (*UPIPaymentResponse, error)
    CheckStatus(ctx context.Context, transactionID string) (*UPIStatusResponse, error)
    GenerateQR(ctx context.Context, req QRRequest) (*QRResponse, error)
}

func (upp *UPIPaymentProcessor) ProcessPayment(ctx context.Context, req PaymentRequest) (*PaymentResponse, error) {
    upiDetails := req.PaymentMethod.UPI
    
    // Validate VPA format
    if err := upp.vpaValidator.ValidateVPA(upiDetails.VPA); err != nil {
        return nil, fmt.Errorf("invalid VPA: %w", err)
    }
    
    // Create UPI payment request
    upiReq := UPIPaymentRequest{
        PayerVPA:     upiDetails.VPA,
        PayeeVPA:     upp.getMerchantVPA(req.MerchantID),
        Amount:       req.Amount,
        Currency:     req.Currency,
        Reference:    req.PaymentID,
        Description:  req.OrderDetails.Description,
        MerchantCode: req.MerchantID,
    }
    
    // Select UPI gateway
    gateway := upp.selectUPIGateway(upiDetails)
    
    // Initiate payment
    upiResp, err := gateway.InitiatePayment(ctx, upiReq)
    if err != nil {
        return nil, fmt.Errorf("UPI payment initiation failed: %w", err)
    }
    
    // For UPI, we typically get async response
    return &PaymentResponse{
        PaymentID:     req.PaymentID,
        Status:        PaymentStatusCreated,
        TransactionID: upiResp.TransactionID,
        UPIResponse:   upiResp,
        ProcessedAt:   time.Now(),
    }, nil
}

// Payment Settlement Engine
type SettlementEngine struct {
    settlementRules  []SettlementRule
    bankingPartners  map[string]BankingPartner
    schedulingService *SchedulingService
    reconciliationService *ReconciliationService
    settlementDB     SettlementRepository
}

type SettlementRule struct {
    MerchantTier    string        `json:"merchant_tier"`
    PaymentMethod   PaymentType   `json:"payment_method"`
    SettlementDelay time.Duration `json:"settlement_delay"`
    HoldPercentage  float64       `json:"hold_percentage"`
    Fees            FeeStructure  `json:"fees"`
}

type FeeStructure struct {
    FixedFee      int64   `json:"fixed_fee"`      // in paise
    PercentageFee float64 `json:"percentage_fee"` // percentage
    GST           float64 `json:"gst"`            // GST percentage
}

func (se *SettlementEngine) ScheduleSettlement(ctx context.Context, payment *PaymentResponse) error {
    // Determine settlement rules
    rule := se.getSettlementRule(payment.MerchantID, payment.PaymentMethod)
    
    // Calculate settlement amount
    settlementAmount := se.calculateSettlementAmount(payment.ProcessedAmount, rule.Fees)
    
    // Create settlement record
    settlement := &Settlement{
        PaymentID:       payment.PaymentID,
        MerchantID:      payment.MerchantID,
        Amount:          payment.ProcessedAmount,
        SettlementAmount: settlementAmount,
        Fees:            rule.Fees,
        ScheduledAt:     time.Now().Add(rule.SettlementDelay),
        Status:          SettlementStatusScheduled,
        PaymentMethod:   payment.PaymentMethod,
    }
    
    // Save to database
    if err := se.settlementDB.CreateSettlement(ctx, settlement); err != nil {
        return fmt.Errorf("failed to create settlement record: %w", err)
    }
    
    // Schedule for processing
    return se.schedulingService.ScheduleTask(ctx, ScheduledTask{
        Type:        TaskTypeSettlement,
        ScheduledAt: settlement.ScheduledAt,
        Data:        settlement.ID,
    })
}

func (se *SettlementEngine) ProcessScheduledSettlements(ctx context.Context) error {
    // Get due settlements
    settlements, err := se.settlementDB.GetDueSettlements(ctx, time.Now())
    if err != nil {
        return fmt.Errorf("failed to fetch due settlements: %w", err)
    }
    
    // Process in batches
    batchSize := 1000
    for i := 0; i < len(settlements); i += batchSize {
        end := i + batchSize
        if end > len(settlements) {
            end = len(settlements)
        }
        
        batch := settlements[i:end]
        if err := se.processBatch(ctx, batch); err != nil {
            // Log error but continue with other batches
            continue
        }
    }
    
    return nil
}

func (se *SettlementEngine) processBatch(ctx context.Context, settlements []*Settlement) error {
    // Group by merchant and banking partner
    merchantBatches := se.groupByMerchant(settlements)
    
    for merchantID, merchantSettlements := range merchantBatches {
        // Get merchant banking details
        bankingDetails, err := se.getMerchantBankingDetails(merchantID)
        if err != nil {
            continue
        }
        
        // Create bank transfer
        totalAmount := se.calculateTotalAmount(merchantSettlements)
        
        transfer := BankTransfer{
            MerchantID:    merchantID,
            Amount:        totalAmount,
            BankAccount:   bankingDetails.Account,
            Reference:     fmt.Sprintf("RZP-SETTLEMENT-%s", time.Now().Format("20060102")),
            Settlements:   merchantSettlements,
        }
        
        // Execute transfer
        if err := se.executeBankTransfer(ctx, transfer); err != nil {
            // Mark settlements as failed
            se.markSettlementsAsFailed(ctx, merchantSettlements, err)
            continue
        }
        
        // Mark settlements as processed
        se.markSettlementsAsProcessed(ctx, merchantSettlements)
    }
    
    return nil
}
```

---

## 🛡️ **Financial Compliance & Security**

### **Regulatory Compliance Framework**

```go
package compliance

import (
    "context"
    "fmt"
    "time"
)

// Compliance Engine for regulatory requirements
type ComplianceEngine struct {
    amlService      *AMLService
    kycService      *KYCService
    pciCompliance   *PCIComplianceService
    dataProtection  *DataProtectionService
    auditTrail      *AuditTrail
    reportingService *RegulatoryReportingService
    
    // Indian regulatory requirements
    rbiCompliance   *RBIComplianceService
    gstCompliance   *GSTComplianceService
    femaCompliance  *FEMAComplianceService
}

// Anti-Money Laundering (AML) Service
type AMLService struct {
    transactionMonitor *TransactionMonitor
    sanctionsList      *SanctionsListService
    suspiciousActivityDetector *SuspiciousActivityDetector
    caseManagement     *CaseManagementSystem
}

type TransactionMonitor struct {
    rules           []AMLRule
    velocityChecker *VelocityChecker
    patternDetector *PatternDetector
    alertSystem     *AlertSystem
}

type AMLRule struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Description string                 `json:"description"`
    Condition   AMLCondition          `json:"condition"`
    Action      AMLAction             `json:"action"`
    Severity    AlertSeverity         `json:"severity"`
    Active      bool                  `json:"active"`
    Metadata    map[string]interface{} `json:"metadata"`
}

type AMLCondition struct {
    AmountThreshold    *int64        `json:"amount_threshold,omitempty"`
    FrequencyThreshold *int          `json:"frequency_threshold,omitempty"`
    TimeWindow         *time.Duration `json:"time_window,omitempty"`
    CountryRestriction []string      `json:"country_restriction,omitempty"`
    MerchantCategory   []string      `json:"merchant_category,omitempty"`
    PaymentPattern     *PatternRule  `json:"payment_pattern,omitempty"`
}

type AMLAction string

const (
    AMLActionAlert      AMLAction = "alert"
    AMLActionBlock      AMLAction = "block"
    AMLActionReview     AMLAction = "review"
    AMLActionReport     AMLAction = "report"
    AMLActionEscalate   AMLAction = "escalate"
)

func (aml *AMLService) AnalyzeTransaction(ctx context.Context, txn Transaction) (*AMLResult, error) {
    result := &AMLResult{
        TransactionID: txn.ID,
        Timestamp:     time.Now(),
        Alerts:        []AMLAlert{},
    }
    
    // Check sanctions list
    if violation := aml.sanctionsList.CheckViolation(txn.Customer); violation != nil {
        result.Alerts = append(result.Alerts, AMLAlert{
            Type:        AlertTypeSanctions,
            Severity:    AlertSeverityHigh,
            Description: fmt.Sprintf("Customer on sanctions list: %s", violation.Reason),
            Action:      AMLActionBlock,
        })
        result.RiskScore += 100 // Maximum risk
    }
    
    // Check transaction patterns
    patterns := aml.transactionMonitor.patternDetector.DetectPatterns(txn)
    for _, pattern := range patterns {
        if pattern.Suspicious {
            result.Alerts = append(result.Alerts, AMLAlert{
                Type:        AlertTypePattern,
                Severity:    pattern.Severity,
                Description: pattern.Description,
                Action:      pattern.RecommendedAction,
            })
            result.RiskScore += pattern.RiskScore
        }
    }
    
    // Velocity checks
    velocityResult := aml.transactionMonitor.velocityChecker.CheckVelocity(txn)
    if velocityResult.Exceeded {
        result.Alerts = append(result.Alerts, AMLAlert{
            Type:        AlertTypeVelocity,
            Severity:    AlertSeverityMedium,
            Description: fmt.Sprintf("Velocity limit exceeded: %s", velocityResult.Description),
            Action:      AMLActionReview,
        })
        result.RiskScore += velocityResult.RiskScore
    }
    
    // Apply AML rules
    for _, rule := range aml.transactionMonitor.rules {
        if rule.Active && aml.evaluateRule(rule, txn) {
            result.Alerts = append(result.Alerts, AMLAlert{
                Type:        AlertTypeRule,
                Severity:    rule.Severity,
                Description: fmt.Sprintf("AML rule triggered: %s", rule.Name),
                Action:      rule.Action,
                RuleID:      rule.ID,
            })
            result.RiskScore += aml.getRuleRiskScore(rule)
        }
    }
    
    // Determine final action
    result.FinalAction = aml.determineFinalAction(result)
    
    // Create case if high risk
    if result.RiskScore > 70 {
        caseID, err := aml.caseManagement.CreateCase(ctx, CaseRequest{
            TransactionID: txn.ID,
            RiskScore:     result.RiskScore,
            Alerts:        result.Alerts,
            Priority:      aml.calculateCasePriority(result.RiskScore),
        })
        if err != nil {
            return nil, fmt.Errorf("failed to create AML case: %w", err)
        }
        result.CaseID = caseID
    }
    
    return result, nil
}

// KYC (Know Your Customer) Service
type KYCService struct {
    documentVerifier *DocumentVerifier
    identityVerifier *IdentityVerifier
    addressVerifier  *AddressVerifier
    businessVerifier *BusinessVerifier
    onboardingWorkflow *OnboardingWorkflow
}

type KYCStatus string

const (
    KYCStatusPending    KYCStatus = "pending"
    KYCStatusInProgress KYCStatus = "in_progress"
    KYCStatusVerified   KYCStatus = "verified"
    KYCStatusRejected   KYCStatus = "rejected"
    KYCStatusExpired    KYCStatus = "expired"
)

type MerchantKYC struct {
    MerchantID       string                 `json:"merchant_id"`
    Status           KYCStatus              `json:"status"`
    DocumentsSubmitted []KYCDocument       `json:"documents_submitted"`
    VerificationSteps  []VerificationStep   `json:"verification_steps"`
    RiskRating       RiskRating             `json:"risk_rating"`
    ComplianceChecks []ComplianceCheck      `json:"compliance_checks"`
    LastUpdated      time.Time              `json:"last_updated"`
    ExpiryDate       time.Time              `json:"expiry_date"`
    Metadata         map[string]interface{} `json:"metadata"`
}

type KYCDocument struct {
    Type         DocumentType `json:"type"`
    Number       string       `json:"number"`
    IssuedDate   time.Time    `json:"issued_date"`
    ExpiryDate   time.Time    `json:"expiry_date"`
    VerificationStatus DocumentVerificationStatus `json:"verification_status"`
    UploadedFile string      `json:"uploaded_file"`
    VerifiedAt   time.Time   `json:"verified_at"`
}

type DocumentType string

const (
    DocumentTypePAN       DocumentType = "pan"
    DocumentTypeAadhar    DocumentType = "aadhar"
    DocumentTypeGST       DocumentType = "gst"
    DocumentTypeBankProof DocumentType = "bank_proof"
    DocumentTypeBusinessReg DocumentType = "business_registration"
    DocumentTypeAddressProof DocumentType = "address_proof"
)

func (kyc *KYCService) InitiateMerchantKYC(ctx context.Context, merchantID string, documents []KYCDocument) error {
    // Create KYC record
    merchantKYC := &MerchantKYC{
        MerchantID:         merchantID,
        Status:             KYCStatusInProgress,
        DocumentsSubmitted: documents,
        VerificationSteps:  []VerificationStep{},
        LastUpdated:        time.Now(),
    }
    
    // Start verification workflow
    workflow := kyc.onboardingWorkflow.CreateWorkflow(merchantID)
    
    // Document verification
    for _, doc := range documents {
        step := VerificationStep{
            Type:      VerificationTypeDocument,
            Status:    VerificationStatusPending,
            StartedAt: time.Now(),
            Document:  &doc,
        }
        
        // Verify document
        verificationResult, err := kyc.documentVerifier.VerifyDocument(ctx, doc)
        if err != nil {
            step.Status = VerificationStatusFailed
            step.Error = err.Error()
        } else {
            step.Status = verificationResult.Status
            step.VerificationResult = verificationResult
        }
        
        step.CompletedAt = time.Now()
        merchantKYC.VerificationSteps = append(merchantKYC.VerificationSteps, step)
    }
    
    // Identity verification
    identityStep := VerificationStep{
        Type:      VerificationTypeIdentity,
        Status:    VerificationStatusPending,
        StartedAt: time.Now(),
    }
    
    identityResult, err := kyc.identityVerifier.VerifyIdentity(ctx, merchantID, documents)
    if err != nil {
        identityStep.Status = VerificationStatusFailed
        identityStep.Error = err.Error()
    } else {
        identityStep.Status = identityResult.Status
        identityStep.VerificationResult = identityResult
    }
    
    identityStep.CompletedAt = time.Now()
    merchantKYC.VerificationSteps = append(merchantKYC.VerificationSteps, identityStep)
    
    // Business verification (if applicable)
    if kyc.isBusinessMerchant(documents) {
        businessStep := VerificationStep{
            Type:      VerificationTypeBusiness,
            Status:    VerificationStatusPending,
            StartedAt: time.Now(),
        }
        
        businessResult, err := kyc.businessVerifier.VerifyBusiness(ctx, merchantID, documents)
        if err != nil {
            businessStep.Status = VerificationStatusFailed
            businessStep.Error = err.Error()
        } else {
            businessStep.Status = businessResult.Status
            businessStep.VerificationResult = businessResult
        }
        
        businessStep.CompletedAt = time.Now()
        merchantKYC.VerificationSteps = append(merchantKYC.VerificationSteps, businessStep)
    }
    
    // Calculate risk rating
    merchantKYC.RiskRating = kyc.calculateRiskRating(merchantKYC)
    
    // Determine final KYC status
    merchantKYC.Status = kyc.determineFinalStatus(merchantKYC.VerificationSteps)
    
    // Set expiry date
    if merchantKYC.Status == KYCStatusVerified {
        merchantKYC.ExpiryDate = time.Now().AddDate(1, 0, 0) // 1 year validity
    }
    
    // Save KYC record
    return kyc.saveMerchantKYC(ctx, merchantKYC)
}

// PCI DSS Compliance Service
type PCIComplianceService struct {
    tokenVault       *TokenVault
    encryptionService *EncryptionService
    accessController *AccessController
    auditLogger      *AuditLogger
    complianceMonitor *ComplianceMonitor
}

type TokenVault struct {
    encryptionKey    []byte
    tokenGenerator   *TokenGenerator
    tokenStorage     TokenStorage
    detokenizer      *Detokenizer
}

func (pci *PCIComplianceService) TokenizeCardData(cardData CardData) (*TokenizedCard, error) {
    // Validate card data format
    if err := pci.validateCardData(cardData); err != nil {
        return nil, fmt.Errorf("invalid card data: %w", err)
    }
    
    // Generate token
    token, err := pci.tokenVault.tokenGenerator.GenerateToken()
    if err != nil {
        return nil, fmt.Errorf("token generation failed: %w", err)
    }
    
    // Encrypt sensitive data
    encryptedPAN, err := pci.encryptionService.Encrypt(cardData.PAN)
    if err != nil {
        return nil, fmt.Errorf("encryption failed: %w", err)
    }
    
    // Store token mapping
    tokenRecord := &TokenRecord{
        Token:        token,
        EncryptedPAN: encryptedPAN,
        ExpiryMonth:  cardData.ExpiryMonth,
        ExpiryYear:   cardData.ExpiryYear,
        CreatedAt:    time.Now(),
        LastUsed:     time.Now(),
    }
    
    if err := pci.tokenVault.tokenStorage.StoreToken(token, tokenRecord); err != nil {
        return nil, fmt.Errorf("token storage failed: %w", err)
    }
    
    // Audit log
    pci.auditLogger.LogTokenization(token, "card_tokenized")
    
    return &TokenizedCard{
        Token:       token,
        LastFourDigits: cardData.PAN[len(cardData.PAN)-4:],
        ExpiryMonth: cardData.ExpiryMonth,
        ExpiryYear:  cardData.ExpiryYear,
        CreatedAt:   time.Now(),
    }, nil
}
```

This comprehensive guide covers Razorpay's core payment processing architecture, compliance requirements, and system design challenges. The implementations demonstrate production-ready patterns for fintech systems at scale.

---

## ⚡ **Real-Time Processing & Streaming**

### **Event-Driven Payment Architecture**

```go
package streaming

import (
    "context"
    "encoding/json"
    "fmt"
    "time"
)

// Real-time payment event processing system
type PaymentEventProcessor struct {
    kafka           *KafkaCluster
    redis           *RedisCluster
    eventHandlers   map[EventType][]EventHandler
    streamProcessor *StreamProcessor
    cep             *ComplexEventProcessor
    metrics         *StreamingMetrics
}

type PaymentEvent struct {
    EventID       string                 `json:"event_id"`
    EventType     EventType              `json:"event_type"`
    PaymentID     string                 `json:"payment_id"`
    MerchantID    string                 `json:"merchant_id"`
    Timestamp     time.Time              `json:"timestamp"`
    Amount        int64                  `json:"amount"`
    Currency      string                 `json:"currency"`
    Status        PaymentStatus          `json:"status"`
    PaymentMethod PaymentType            `json:"payment_method"`
    Metadata      map[string]interface{} `json:"metadata"`
    Version       int                    `json:"version"`
}

type EventType string

const (
    EventPaymentCreated     EventType = "payment.created"
    EventPaymentAuthorized  EventType = "payment.authorized"
    EventPaymentCaptured    EventType = "payment.captured"
    EventPaymentFailed      EventType = "payment.failed"
    EventPaymentRefunded    EventType = "payment.refunded"
    EventPaymentDisputed    EventType = "payment.disputed"
    EventSettlementCreated  EventType = "settlement.created"
    EventWebhookDelivered   EventType = "webhook.delivered"
    EventFraudDetected      EventType = "fraud.detected"
)

// Real-time fraud detection using streaming
type RealTimeFraudDetector struct {
    eventStream        *EventStream
    ruleEngine         *RuleEngine
    mlModels           *MLModelService
    featureStore       *FeatureStore
    alertPublisher     *AlertPublisher
    decisionEngine     *DecisionEngine
}

func NewRealTimeFraudDetector(config FraudConfig) *RealTimeFraudDetector {
    return &RealTimeFraudDetector{
        eventStream:    NewEventStream(config.StreamConfig),
        ruleEngine:     NewRuleEngine(config.RuleConfig),
        mlModels:       NewMLModelService(config.MLConfig),
        featureStore:   NewFeatureStore(config.FeatureConfig),
        alertPublisher: NewAlertPublisher(config.AlertConfig),
        decisionEngine: NewDecisionEngine(config.DecisionConfig),
    }
}

func (rtfd *RealTimeFraudDetector) ProcessPaymentEvent(ctx context.Context, event PaymentEvent) (*FraudDecision, error) {
    startTime := time.Now()
    
    // Extract features from event and historical data
    features, err := rtfd.featureStore.ExtractFeatures(ctx, event)
    if err != nil {
        return nil, fmt.Errorf("feature extraction failed: %w", err)
    }
    
    // Apply rule-based detection
    ruleResults := rtfd.ruleEngine.EvaluateRules(features)
    
    // Apply ML models for advanced detection
    mlResults, err := rtfd.mlModels.PredictFraud(ctx, features)
    if err != nil {
        // Don't fail on ML errors, continue with rule-based results
        mlResults = &MLPrediction{Score: 0, Confidence: 0}
    }
    
    // Combine results using decision engine
    decision := rtfd.decisionEngine.MakeDecision(ruleResults, mlResults, features)
    
    // Record processing metrics
    processingTime := time.Since(startTime)
    rtfd.recordMetrics(event, decision, processingTime)
    
    // Publish alerts if high risk
    if decision.RiskScore > 70 {
        alert := FraudAlert{
            PaymentID:   event.PaymentID,
            MerchantID:  event.MerchantID,
            RiskScore:   decision.RiskScore,
            Reasons:     decision.Reasons,
            Action:      decision.Action,
            Timestamp:   time.Now(),
        }
        
        if err := rtfd.alertPublisher.PublishAlert(ctx, alert); err != nil {
            // Log error but don't fail the fraud detection
            fmt.Printf("Failed to publish fraud alert: %v\n", err)
        }
    }
    
    return decision, nil
}

// Feature extraction for real-time fraud detection
type FeatureStore struct {
    redis           *RedisCluster
    timeSeriesDB    *InfluxDBClient
    featureCache    *FeatureCache
    featureEngines  map[string]FeatureEngine
}

func (fs *FeatureStore) ExtractFeatures(ctx context.Context, event PaymentEvent) (*FeatureVector, error) {
    features := &FeatureVector{
        PaymentID:  event.PaymentID,
        MerchantID: event.MerchantID,
        Timestamp:  event.Timestamp,
        Features:   make(map[string]interface{}),
    }
    
    // Basic payment features
    features.Features["amount"] = event.Amount
    features.Features["payment_method"] = event.PaymentMethod
    features.Features["hour_of_day"] = event.Timestamp.Hour()
    features.Features["day_of_week"] = int(event.Timestamp.Weekday())
    
    // Velocity features (using Redis for real-time counters)
    velocityFeatures, err := fs.extractVelocityFeatures(ctx, event)
    if err != nil {
        return nil, err
    }
    for k, v := range velocityFeatures {
        features.Features[k] = v
    }
    
    // Historical behavior features
    historicalFeatures, err := fs.extractHistoricalFeatures(ctx, event)
    if err != nil {
        return nil, err
    }
    for k, v := range historicalFeatures {
        features.Features[k] = v
    }
    
    // Merchant-specific features
    merchantFeatures, err := fs.extractMerchantFeatures(ctx, event.MerchantID)
    if err != nil {
        return nil, err
    }
    for k, v := range merchantFeatures {
        features.Features[k] = v
    }
    
    return features, nil
}

func (fs *FeatureStore) extractVelocityFeatures(ctx context.Context, event PaymentEvent) (map[string]interface{}, error) {
    features := make(map[string]interface{})
    now := time.Now()
    
    // Transaction count in last hour
    hourKey := fmt.Sprintf("txn_count:merchant:%s:hour:%s", 
        event.MerchantID, now.Format("2006010215"))
    hourCount, _ := fs.redis.Incr(ctx, hourKey).Result()
    fs.redis.Expire(ctx, hourKey, time.Hour)
    features["txn_count_last_hour"] = hourCount
    
    // Transaction count in last day
    dayKey := fmt.Sprintf("txn_count:merchant:%s:day:%s", 
        event.MerchantID, now.Format("20060102"))
    dayCount, _ := fs.redis.Incr(ctx, dayKey).Result()
    fs.redis.Expire(ctx, dayKey, 24*time.Hour)
    features["txn_count_last_day"] = dayCount
    
    // Amount velocity
    amountHourKey := fmt.Sprintf("amount_sum:merchant:%s:hour:%s", 
        event.MerchantID, now.Format("2006010215"))
    amountHour, _ := fs.redis.IncrBy(ctx, amountHourKey, event.Amount).Result()
    fs.redis.Expire(ctx, amountHourKey, time.Hour)
    features["amount_sum_last_hour"] = amountHour
    
    // Unique customers
    if customerID, ok := event.Metadata["customer_id"]; ok {
        customerKey := fmt.Sprintf("unique_customers:merchant:%s:day:%s", 
            event.MerchantID, now.Format("20060102"))
        fs.redis.SAdd(ctx, customerKey, customerID)
        fs.redis.Expire(ctx, customerKey, 24*time.Hour)
        uniqueCustomers, _ := fs.redis.SCard(ctx, customerKey).Result()
        features["unique_customers_last_day"] = uniqueCustomers
    }
    
    return features, nil
}

// Stream processing for real-time analytics
type StreamProcessor struct {
    kafkaConsumer   *KafkaConsumer
    processors      map[string]Processor
    outputSinks     []OutputSink
    stateStore      StateStore
    checkpointer    Checkpointer
}

type Processor interface {
    Process(ctx context.Context, event Event) ([]Event, error)
    GetName() string
}

// Real-time payment analytics processor
type PaymentAnalyticsProcessor struct {
    metricsStore    *MetricsStore
    alertThresholds map[string]Threshold
    dashboardUpdater *DashboardUpdater
}

func (pap *PaymentAnalyticsProcessor) Process(ctx context.Context, event Event) ([]Event, error) {
    paymentEvent := event.(PaymentEvent)
    
    // Update real-time metrics
    metrics := []Metric{
        {
            Name:      "payment_volume",
            Value:     float64(paymentEvent.Amount),
            Tags:      map[string]string{"merchant_id": paymentEvent.MerchantID},
            Timestamp: paymentEvent.Timestamp,
        },
        {
            Name:      "payment_count",
            Value:     1,
            Tags:      map[string]string{"status": string(paymentEvent.Status)},
            Timestamp: paymentEvent.Timestamp,
        },
    }
    
    for _, metric := range metrics {
        if err := pap.metricsStore.Record(ctx, metric); err != nil {
            return nil, fmt.Errorf("failed to record metric: %w", err)
        }
    }
    
    // Check for threshold violations
    for metricName, threshold := range pap.alertThresholds {
        currentValue, err := pap.metricsStore.GetCurrentValue(ctx, metricName)
        if err != nil {
            continue
        }
        
        if currentValue > threshold.Value {
            alertEvent := AlertEvent{
                EventID:     generateEventID(),
                EventType:   EventTypeThresholdAlert,
                MetricName:  metricName,
                CurrentValue: currentValue,
                Threshold:   threshold.Value,
                Timestamp:   time.Now(),
            }
            
            return []Event{alertEvent}, nil
        }
    }
    
    // Update dashboard
    go pap.dashboardUpdater.UpdateRealTimeMetrics(paymentEvent)
    
    return nil, nil
}

// WebSocket service for real-time merchant dashboard
type RealTimeDashboardService struct {
    wsHub           *WebSocketHub
    eventSubscriber *EventSubscriber
    authService     *AuthService
    metricsAggregator *MetricsAggregator
}

type WebSocketHub struct {
    clients    map[string]*WebSocketClient
    register   chan *WebSocketClient
    unregister chan *WebSocketClient
    broadcast  chan []byte
    mu         sync.RWMutex
}

func (rtds *RealTimeDashboardService) HandleWebSocketConnection(w http.ResponseWriter, r *http.Request) {
    // Authenticate merchant
    merchantID, err := rtds.authService.AuthenticateWebSocket(r)
    if err != nil {
        http.Error(w, "Authentication failed", http.StatusUnauthorized)
        return
    }
    
    // Upgrade to WebSocket
    conn, err := websocket.Upgrade(w, r, nil)
    if err != nil {
        return
    }
    
    client := &WebSocketClient{
        hub:        rtds.wsHub,
        conn:       conn,
        send:       make(chan []byte, 256),
        merchantID: merchantID,
    }
    
    rtds.wsHub.register <- client
    
    // Subscribe to merchant-specific events
    eventChan := rtds.eventSubscriber.Subscribe(fmt.Sprintf("merchant:%s", merchantID))
    
    go client.writePump()
    go client.readPump()
    go rtds.forwardEvents(client, eventChan)
}

func (rtds *RealTimeDashboardService) forwardEvents(client *WebSocketClient, eventChan <-chan Event) {
    for event := range eventChan {
        // Filter events relevant to merchant
        if rtds.isRelevantEvent(event, client.merchantID) {
            // Transform event for dashboard
            dashboardEvent := rtds.transformEvent(event)
            
            // Send to client
            eventJSON, _ := json.Marshal(dashboardEvent)
            select {
            case client.send <- eventJSON:
            default:
                close(client.send)
                return
            }
        }
    }
}
```

---

## 🔍 **Fraud Detection & Risk Management**

### **Advanced Fraud Detection System**

```go
package fraud

import (
    "context"
    "encoding/json"
    "fmt"
    "math"
    "time"
)

// Multi-layered fraud detection system
type FraudDetectionSystem struct {
    ruleEngine      *RuleBasedEngine
    mlEngine        *MLBasedEngine
    behavioralEngine *BehavioralAnalysisEngine
    networkAnalysis  *NetworkAnalysisEngine
    deviceFingerprinting *DeviceFingerprintService
    
    riskScorer      *RiskScorer
    decisionEngine  *FraudDecisionEngine
    caseManagement  *FraudCaseManagement
    feedbackLoop    *FeedbackLoop
}

// Rule-based fraud detection engine
type RuleBasedEngine struct {
    rules          []FraudRule
    ruleEvaluator  *RuleEvaluator
    ruleManager    *RuleManager
    blacklists     *BlacklistService
    whitelists     *WhitelistService
}

type FraudRule struct {
    ID          string        `json:"id"`
    Name        string        `json:"name"`
    Description string        `json:"description"`
    Category    RuleCategory  `json:"category"`
    Conditions  []Condition   `json:"conditions"`
    Action      RuleAction    `json:"action"`
    Weight      float64       `json:"weight"`
    Active      bool          `json:"active"`
    CreatedAt   time.Time     `json:"created_at"`
    UpdatedAt   time.Time     `json:"updated_at"`
}

type RuleCategory string

const (
    RuleCategoryVelocity     RuleCategory = "velocity"
    RuleCategoryAmount       RuleCategory = "amount"
    RuleCategoryGeolocation  RuleCategory = "geolocation"
    RuleCategoryDevice       RuleCategory = "device"
    RuleCategoryBehavioral   RuleCategory = "behavioral"
    RuleCategoryNetwork      RuleCategory = "network"
)

type Condition struct {
    Field     string      `json:"field"`
    Operator  Operator    `json:"operator"`
    Value     interface{} `json:"value"`
    TimeWindow *TimeWindow `json:"time_window,omitempty"`
}

type Operator string

const (
    OperatorEquals         Operator = "equals"
    OperatorNotEquals      Operator = "not_equals"
    OperatorGreaterThan    Operator = "greater_than"
    OperatorLessThan       Operator = "less_than"
    OperatorContains       Operator = "contains"
    OperatorInList         Operator = "in_list"
    OperatorNotInList      Operator = "not_in_list"
    OperatorRegexMatch     Operator = "regex_match"
    OperatorFrequencyExceeds Operator = "frequency_exceeds"
)

// Behavioral analysis for fraud detection
type BehavioralAnalysisEngine struct {
    profileStore     *BehavioralProfileStore
    patternDetector  *PatternDetector
    anomalyDetector  *AnomalyDetector
    sessionAnalyzer  *SessionAnalyzer
}

type BehavioralProfile struct {
    MerchantID      string                 `json:"merchant_id"`
    CustomerID      string                 `json:"customer_id,omitempty"`
    PaymentPatterns PaymentPatterns        `json:"payment_patterns"`
    TimePatterns    TimePatterns           `json:"time_patterns"`
    AmountPatterns  AmountPatterns         `json:"amount_patterns"`
    DevicePatterns  DevicePatterns         `json:"device_patterns"`
    LocationPatterns LocationPatterns      `json:"location_patterns"`
    LastUpdated     time.Time              `json:"last_updated"`
    Confidence      float64                `json:"confidence"`
}

type PaymentPatterns struct {
    PreferredMethods    []PaymentType `json:"preferred_methods"`
    MethodFrequency     map[PaymentType]int `json:"method_frequency"`
    AverageAmount       float64       `json:"average_amount"`
    TypicalAmountRange  AmountRange   `json:"typical_amount_range"`
    FrequencyPattern    FrequencyPattern `json:"frequency_pattern"`
}

func (bae *BehavioralAnalysisEngine) AnalyzePayment(ctx context.Context, payment PaymentRequest) (*BehavioralAnalysis, error) {
    analysis := &BehavioralAnalysis{
        PaymentID: payment.PaymentID,
        Timestamp: time.Now(),
        Anomalies: []BehavioralAnomaly{},
    }
    
    // Get or create behavioral profile
    profile, err := bae.profileStore.GetProfile(ctx, payment.MerchantID, payment.CustomerDetails.CustomerID)
    if err != nil && err != ErrProfileNotFound {
        return nil, fmt.Errorf("failed to get behavioral profile: %w", err)
    }
    
    if profile == nil {
        // New customer - create baseline profile
        profile = bae.createBaselineProfile(payment)
        analysis.IsNewCustomer = true
    }
    
    // Analyze payment against profile
    anomalies := bae.detectAnomalies(payment, profile)
    analysis.Anomalies = append(analysis.Anomalies, anomalies...)
    
    // Pattern analysis
    patterns := bae.patternDetector.DetectPatterns(payment, profile)
    analysis.PatternMatches = patterns
    
    // Session analysis
    sessionAnalysis := bae.sessionAnalyzer.AnalyzeSession(ctx, payment)
    analysis.SessionRisk = sessionAnalysis.RiskScore
    
    // Calculate behavioral risk score
    analysis.BehavioralRiskScore = bae.calculateBehavioralRisk(analysis)
    
    // Update profile with new payment data
    updatedProfile := bae.updateProfile(profile, payment)
    if err := bae.profileStore.SaveProfile(ctx, updatedProfile); err != nil {
        // Log error but don't fail analysis
        fmt.Printf("Failed to update behavioral profile: %v\n", err)
    }
    
    return analysis, nil
}

func (bae *BehavioralAnalysisEngine) detectAnomalies(payment PaymentRequest, profile *BehavioralProfile) []BehavioralAnomaly {
    var anomalies []BehavioralAnomaly
    
    // Amount anomaly detection
    if bae.isAmountAnomaly(payment.Amount, profile.AmountPatterns) {
        anomalies = append(anomalies, BehavioralAnomaly{
            Type:        AnomalyTypeAmount,
            Description: fmt.Sprintf("Amount %d is unusual for this customer", payment.Amount),
            Severity:    bae.calculateAmountAnomalySeverity(payment.Amount, profile.AmountPatterns),
            Confidence:  0.8,
        })
    }
    
    // Time pattern anomaly
    currentHour := time.Now().Hour()
    if bae.isTimeAnomaly(currentHour, profile.TimePatterns) {
        anomalies = append(anomalies, BehavioralAnomaly{
            Type:        AnomalyTypeTime,
            Description: fmt.Sprintf("Payment at hour %d is unusual for this customer", currentHour),
            Severity:    AnomalySeverityMedium,
            Confidence:  0.7,
        })
    }
    
    // Payment method anomaly
    if bae.isPaymentMethodAnomaly(payment.PaymentMethod.Type, profile.PaymentPatterns) {
        anomalies = append(anomalies, BehavioralAnomaly{
            Type:        AnomalyTypePaymentMethod,
            Description: fmt.Sprintf("Payment method %s is unusual for this customer", payment.PaymentMethod.Type),
            Severity:    AnomalySeverityLow,
            Confidence:  0.6,
        })
    }
    
    return anomalies
}

// Machine Learning based fraud detection
type MLBasedEngine struct {
    models          map[string]MLModel
    featureEngineering *FeatureEngineering
    modelService    *ModelService
    predictionCache *PredictionCache
}

type MLModel interface {
    Predict(ctx context.Context, features FeatureVector) (*MLPrediction, error)
    GetModelInfo() ModelInfo
    IsHealthy() bool
}

type ModelInfo struct {
    Name        string    `json:"name"`
    Version     string    `json:"version"`
    Algorithm   string    `json:"algorithm"`
    TrainedAt   time.Time `json:"trained_at"`
    Accuracy    float64   `json:"accuracy"`
    Precision   float64   `json:"precision"`
    Recall      float64   `json:"recall"`
    F1Score     float64   `json:"f1_score"`
}

// Ensemble model combining multiple ML algorithms
type EnsembleModel struct {
    models      []MLModel
    weights     []float64
    combiner    *ModelCombiner
    validator   *ModelValidator
}

func (em *EnsembleModel) Predict(ctx context.Context, features FeatureVector) (*MLPrediction, error) {
    predictions := make([]*MLPrediction, len(em.models))
    
    // Get predictions from all models in parallel
    var wg sync.WaitGroup
    for i, model := range em.models {
        wg.Add(1)
        go func(index int, m MLModel) {
            defer wg.Done()
            pred, err := m.Predict(ctx, features)
            if err != nil {
                // Use default prediction if model fails
                pred = &MLPrediction{Score: 0.5, Confidence: 0.0}
            }
            predictions[index] = pred
        }(i, model)
    }
    
    wg.Wait()
    
    // Combine predictions using weighted average
    combinedScore := 0.0
    combinedConfidence := 0.0
    totalWeight := 0.0
    
    for i, pred := range predictions {
        weight := em.weights[i]
        combinedScore += pred.Score * weight
        combinedConfidence += pred.Confidence * weight
        totalWeight += weight
    }
    
    if totalWeight > 0 {
        combinedScore /= totalWeight
        combinedConfidence /= totalWeight
    }
    
    return &MLPrediction{
        Score:      combinedScore,
        Confidence: combinedConfidence,
        ModelInfo:  "ensemble",
        Features:   features,
        Timestamp:  time.Now(),
    }, nil
}

// Network analysis for fraud detection
type NetworkAnalysisEngine struct {
    graphDB         *GraphDatabase
    communityDetector *CommunityDetector
    linkAnalyzer    *LinkAnalyzer
    velocityAnalyzer *NetworkVelocityAnalyzer
}

type PaymentGraph struct {
    Nodes []GraphNode `json:"nodes"`
    Edges []GraphEdge `json:"edges"`
}

type GraphNode struct {
    ID       string                            `json:"id"`
    Type     NodeType                          `json:"type"`
    Labels   []string                          `json:"labels"`
    Properties map[string]interface{}         `json:"properties"`
}

type GraphEdge struct {
    Source     string                         `json:"source"`
    Target     string                         `json:"target"`
    Type       EdgeType                       `json:"type"`
    Weight     float64                        `json:"weight"`
    Properties map[string]interface{}         `json:"properties"`
    Timestamp  time.Time                      `json:"timestamp"`
}

func (nae *NetworkAnalysisEngine) AnalyzePaymentNetwork(ctx context.Context, payment PaymentRequest) (*NetworkAnalysis, error) {
    analysis := &NetworkAnalysis{
        PaymentID: payment.PaymentID,
        Timestamp: time.Now(),
    }
    
    // Build payment graph
    graph, err := nae.buildPaymentGraph(ctx, payment)
    if err != nil {
        return nil, fmt.Errorf("failed to build payment graph: %w", err)
    }
    
    // Community detection
    communities := nae.communityDetector.DetectCommunities(graph)
    analysis.Communities = communities
    
    // Analyze suspicious connections
    suspiciousLinks := nae.linkAnalyzer.FindSuspiciousLinks(graph)
    analysis.SuspiciousConnections = suspiciousLinks
    
    // Network velocity analysis
    velocityRisk := nae.velocityAnalyzer.AnalyzeVelocity(ctx, payment, graph)
    analysis.VelocityRisk = velocityRisk
    
    // Calculate network risk score
    analysis.NetworkRiskScore = nae.calculateNetworkRisk(analysis)
    
    return analysis, nil
}

// Device fingerprinting service
type DeviceFingerprintService struct {
    fingerprintStore *FingerprintStore
    deviceProfiler   *DeviceProfiler
    anomalyDetector  *DeviceAnomalyDetector
}

type DeviceFingerprint struct {
    FingerprintID   string                 `json:"fingerprint_id"`
    UserAgent       string                 `json:"user_agent"`
    IPAddress       string                 `json:"ip_address"`
    ScreenResolution string                `json:"screen_resolution"`
    Timezone        string                 `json:"timezone"`
    Languages       []string               `json:"languages"`
    Plugins         []string               `json:"plugins"`
    Fonts           []string               `json:"fonts"`
    Canvas          string                 `json:"canvas"`
    WebGL           string                 `json:"webgl"`
    AudioContext    string                 `json:"audio_context"`
    BatteryInfo     *BatteryInfo           `json:"battery_info,omitempty"`
    NetworkInfo     *NetworkInfo           `json:"network_info,omitempty"`
    DeviceMotion    *DeviceMotionInfo      `json:"device_motion,omitempty"`
    CustomFields    map[string]interface{} `json:"custom_fields"`
    FirstSeen       time.Time              `json:"first_seen"`
    LastSeen        time.Time              `json:"last_seen"`
    RiskScore       float64                `json:"risk_score"`
}

func (dfs *DeviceFingerprintService) AnalyzeDevice(ctx context.Context, request *http.Request, payment PaymentRequest) (*DeviceAnalysis, error) {
    // Extract device fingerprint
    fingerprint := dfs.extractFingerprint(request)
    
    // Check if device is known
    existingDevice, err := dfs.fingerprintStore.GetDevice(ctx, fingerprint.FingerprintID)
    if err != nil && err != ErrDeviceNotFound {
        return nil, fmt.Errorf("failed to get device: %w", err)
    }
    
    analysis := &DeviceAnalysis{
        PaymentID:   payment.PaymentID,
        Fingerprint: fingerprint,
        IsNewDevice: existingDevice == nil,
        Timestamp:   time.Now(),
    }
    
    if existingDevice != nil {
        // Analyze device behavior changes
        changes := dfs.detectFingerprintChanges(fingerprint, existingDevice)
        analysis.FingerprintChanges = changes
        
        // Check for device reputation
        reputation := dfs.getDeviceReputation(ctx, existingDevice)
        analysis.DeviceReputation = reputation
        
        // Update device profile
        updatedDevice := dfs.updateDeviceProfile(existingDevice, fingerprint, payment)
        dfs.fingerprintStore.SaveDevice(ctx, updatedDevice)
    } else {
        // New device - create profile
        newDevice := dfs.createDeviceProfile(fingerprint, payment)
        dfs.fingerprintStore.SaveDevice(ctx, newDevice)
    }
    
    // Anomaly detection
    anomalies := dfs.anomalyDetector.DetectAnomalies(fingerprint, payment)
    analysis.Anomalies = anomalies
    
    // Calculate device risk score
    analysis.DeviceRiskScore = dfs.calculateDeviceRisk(analysis)
    
    return analysis, nil
}
```

---

## 🎯 **Razorpay Interview Questions & Scenarios**

### **System Design Questions**

**Q1: Design Razorpay's payment gateway to handle 50,000 TPS during peak hours.**

**Answer Approach:**
```
1. Architecture Overview:
   - Load balancers with geographic distribution
   - Microservices architecture with domain separation
   - Event-driven architecture using Kafka
   - Multi-region database setup with read replicas

2. Payment Processing Pipeline:
   - Request validation and sanitization
   - Fraud detection (real-time + batch)
   - Payment routing based on success rates
   - Asynchronous settlement processing

3. Scaling Strategies:
   - Horizontal scaling of application servers
   - Database sharding by merchant_id
   - Caching layers (Redis cluster)
   - Circuit breakers for external dependencies

4. High Availability:
   - Multi-AZ deployment
   - Graceful degradation strategies
   - Backup payment processors
   - Real-time monitoring and alerting
```

**Q2: How would you design a real-time fraud detection system for Razorpay?**

**Answer Approach:**
```
1. Multi-layered Detection:
   - Rule-based engine for known patterns
   - ML models for anomaly detection
   - Behavioral analysis for user patterns
   - Network analysis for connected fraud

2. Real-time Processing:
   - Stream processing with Kafka Streams
   - Feature stores for fast feature lookup
   - In-memory scoring engines
   - Sub-100ms decision making

3. Feedback Loop:
   - Manual review queues
   - Model retraining pipelines
   - False positive optimization
   - Continuous model improvement

4. Action Framework:
   - Risk scoring and thresholds
   - Automated blocking vs manual review
   - Merchant notification systems
   - Case management workflows
```

**Q3: Design Razorpay's settlement system for timely merchant payouts.**

**Answer Implementation:**
- Scheduled settlement processing
- Banking partner integration
- Reconciliation and error handling
- Compliance and audit trails

### **Technical Deep Dive Questions**

**Q4: How do you ensure PCI DSS compliance in payment processing?**

**Answer:**
- Card data tokenization and vault
- Encryption at rest and in transit
- Network segmentation and access controls
- Regular security audits and monitoring
- Secure coding practices and reviews

**Q5: Explain your approach to handling payment failures and retries.**

**Answer:**
- Exponential backoff with jitter
- Circuit breaker patterns
- Fallback payment processors
- Dead letter queues for failed payments
- Comprehensive error classification

**Q6: How would you implement idempotency in payment APIs?**

**Answer:**
- Idempotency keys in request headers
- Database constraints for duplicate prevention
- Distributed locking for concurrent requests
- Response caching for repeated requests
- Proper HTTP status code handling

### **Razorpay-Specific Scenarios**

**Q7: A merchant reports payment success rate drop from 95% to 85%. How do you investigate?**

**Investigation Steps:**
1. Check payment routing and processor health
2. Analyze failure patterns by payment method
3. Review fraud detection rule changes
4. Monitor bank/processor notifications
5. Check for system performance issues
6. Review recent deployments and changes

**Q8: During Diwali sale, payment volumes increase 10x. How do you prepare?**

**Preparation Strategy:**
1. Capacity planning and load testing
2. Auto-scaling policies configuration
3. Database connection pool tuning
4. Cache warming and optimization
5. Monitoring dashboard setup
6. Incident response team standby
7. Communication plan with merchants

**Q9: How would you handle a scenario where a major payment processor goes down?**

**Response Plan:**
1. Activate circuit breakers immediately
2. Route traffic to backup processors
3. Communicate with affected merchants
4. Monitor system health metrics
5. Implement temporary rate limiting
6. Plan processor restoration strategy

### **Code Quality & Best Practices**

**Q10: Show how you would implement a payment webhook delivery system.**

**Implementation considerations:**
- Reliable message delivery with retries
- Webhook signature verification
- Exponential backoff for failures
- Dead letter queues for persistent failures
- Monitoring and alerting for webhook health

This comprehensive Razorpay guide covers the critical technical areas, system design patterns, and interview scenarios specific to fintech and payment processing at scale.