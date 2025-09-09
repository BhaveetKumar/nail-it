# Template Method Pattern

## Pattern Name & Intent

**Template Method** is a behavioral design pattern that defines the skeleton of an algorithm in a superclass but lets subclasses override specific steps of the algorithm without changing its structure.

**Key Intent:**
- Define the skeleton of an algorithm in an operation
- Let subclasses redefine certain steps without changing the algorithm's structure
- Promote code reuse through inheritance
- Implement invariant parts of algorithms once
- Control which parts of an algorithm can be customized
- Follow the Hollywood Principle: "Don't call us, we'll call you"

## When to Use

**Use Template Method when:**

1. **Common Algorithm Structure**: Multiple classes have similar algorithms with variations
2. **Code Duplication**: Common behavior exists across multiple classes
3. **Framework Development**: Need to define extension points for clients
4. **Invariant Behavior**: Part of the algorithm should remain constant
5. **Controlled Customization**: Want to control which parts can be overridden
6. **Hook Methods**: Need optional extension points in algorithms
7. **Workflow Definition**: Standard process with customizable steps

**Don't use when:**
- Algorithm has no common structure across implementations
- High flexibility is needed (composition might be better)
- Runtime algorithm switching is required
- The template becomes too complex or rigid

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Processing Template
```go
// Abstract payment processor template
type PaymentProcessor interface {
    ProcessPayment(ctx context.Context, request *PaymentRequest) (*PaymentResult, error)
    
    // Template method - defines the algorithm skeleton
    processPaymentTemplate(ctx context.Context, request *PaymentRequest) (*PaymentResult, error)
    
    // Abstract methods - must be implemented by subclasses
    validatePaymentData(ctx context.Context, request *PaymentRequest) error
    authenticatePayment(ctx context.Context, request *PaymentRequest) error
    executePayment(ctx context.Context, request *PaymentRequest) (*PaymentResponse, error)
    
    // Hook methods - optional overrides
    preProcessHook(ctx context.Context, request *PaymentRequest) error
    postProcessHook(ctx context.Context, result *PaymentResult) error
    onPaymentSuccess(ctx context.Context, result *PaymentResult) error
    onPaymentFailure(ctx context.Context, err error) error
}

// Base payment processor with template method implementation
type BasePaymentProcessor struct {
    logger    *zap.Logger
    metrics   MetricsCollector
    auditor   AuditLogger
    config    PaymentConfig
}

func NewBasePaymentProcessor(logger *zap.Logger, metrics MetricsCollector, auditor AuditLogger, config PaymentConfig) *BasePaymentProcessor {
    return &BasePaymentProcessor{
        logger:  logger,
        metrics: metrics,
        auditor: auditor,
        config:  config,
    }
}

// Template method - defines the payment processing algorithm
func (bpp *BasePaymentProcessor) ProcessPayment(ctx context.Context, request *PaymentRequest) (*PaymentResult, error) {
    return bpp.processPaymentTemplate(ctx, request)
}

func (bpp *BasePaymentProcessor) processPaymentTemplate(ctx context.Context, request *PaymentRequest) (*PaymentResult, error) {
    startTime := time.Now()
    bpp.logger.Info("Starting payment processing", 
        zap.String("payment_id", request.PaymentID),
        zap.String("amount", request.Amount.String()),
        zap.String("currency", request.Currency))
    
    // Step 1: Pre-processing hook (optional)
    if err := bpp.preProcessHook(ctx, request); err != nil {
        bpp.logger.Warn("Pre-processing hook failed", zap.Error(err))
        return nil, fmt.Errorf("pre-processing failed: %w", err)
    }
    
    // Step 2: Validate payment data (must be implemented by subclass)
    if err := bpp.validatePaymentData(ctx, request); err != nil {
        bpp.metrics.IncrementCounter("payment_validation_failed", request.PaymentMethod)
        bpp.onPaymentFailure(ctx, err)
        return nil, fmt.Errorf("validation failed: %w", err)
    }
    
    // Step 3: Authenticate payment (must be implemented by subclass)
    if err := bpp.authenticatePayment(ctx, request); err != nil {
        bpp.metrics.IncrementCounter("payment_authentication_failed", request.PaymentMethod)
        bpp.onPaymentFailure(ctx, err)
        return nil, fmt.Errorf("authentication failed: %w", err)
    }
    
    // Step 4: Execute payment (must be implemented by subclass)
    response, err := bpp.executePayment(ctx, request)
    if err != nil {
        bpp.metrics.IncrementCounter("payment_execution_failed", request.PaymentMethod)
        bpp.onPaymentFailure(ctx, err)
        return nil, fmt.Errorf("execution failed: %w", err)
    }
    
    // Step 5: Create result
    result := &PaymentResult{
        PaymentID:     request.PaymentID,
        TransactionID: response.TransactionID,
        Status:        response.Status,
        Amount:        request.Amount,
        Currency:      request.Currency,
        ProcessedAt:   time.Now(),
        ProcessingTime: time.Since(startTime),
        Gateway:       response.Gateway,
        Metadata:      response.Metadata,
    }
    
    // Step 6: Handle success
    if err := bpp.onPaymentSuccess(ctx, result); err != nil {
        bpp.logger.Warn("Post-success handler failed", zap.Error(err))
        // Don't fail the payment, just log the warning
    }
    
    // Step 7: Post-processing hook (optional)
    if err := bpp.postProcessHook(ctx, result); err != nil {
        bpp.logger.Warn("Post-processing hook failed", zap.Error(err))
        // Don't fail the payment, just log the warning
    }
    
    // Step 8: Record metrics and audit
    bpp.metrics.RecordDuration("payment_processing_duration", time.Since(startTime), request.PaymentMethod)
    bpp.metrics.IncrementCounter("payment_processed_successfully", request.PaymentMethod)
    
    bpp.auditor.LogPaymentProcessed(ctx, request, result)
    
    bpp.logger.Info("Payment processing completed", 
        zap.String("payment_id", request.PaymentID),
        zap.String("transaction_id", result.TransactionID),
        zap.String("status", result.Status),
        zap.Duration("processing_time", result.ProcessingTime))
    
    return result, nil
}

// Default hook implementations (can be overridden)
func (bpp *BasePaymentProcessor) preProcessHook(ctx context.Context, request *PaymentRequest) error {
    // Default: no pre-processing
    return nil
}

func (bpp *BasePaymentProcessor) postProcessHook(ctx context.Context, result *PaymentResult) error {
    // Default: no post-processing
    return nil
}

func (bpp *BasePaymentProcessor) onPaymentSuccess(ctx context.Context, result *PaymentResult) error {
    // Default: log success
    bpp.logger.Debug("Payment processed successfully", 
        zap.String("payment_id", result.PaymentID),
        zap.String("transaction_id", result.TransactionID))
    return nil
}

func (bpp *BasePaymentProcessor) onPaymentFailure(ctx context.Context, err error) error {
    // Default: log failure
    bpp.logger.Warn("Payment processing failed", zap.Error(err))
    return nil
}

// Concrete implementation: Credit Card Processor
type CreditCardProcessor struct {
    *BasePaymentProcessor
    gateway         CreditCardGateway
    fraudDetector   FraudDetector
    tokenizer       CardTokenizer
    validator       CardValidator
}

func NewCreditCardProcessor(base *BasePaymentProcessor, gateway CreditCardGateway, fraudDetector FraudDetector) *CreditCardProcessor {
    return &CreditCardProcessor{
        BasePaymentProcessor: base,
        gateway:             gateway,
        fraudDetector:       fraudDetector,
        tokenizer:           NewCardTokenizer(),
        validator:           NewCardValidator(),
    }
}

// Implement abstract methods
func (ccp *CreditCardProcessor) validatePaymentData(ctx context.Context, request *PaymentRequest) error {
    ccp.logger.Debug("Validating credit card data", zap.String("payment_id", request.PaymentID))
    
    // Validate card number
    if !ccp.validator.IsValidCardNumber(request.PaymentData.CardNumber) {
        return fmt.Errorf("invalid card number")
    }
    
    // Validate expiry date
    if !ccp.validator.IsValidExpiryDate(request.PaymentData.ExpiryMonth, request.PaymentData.ExpiryYear) {
        return fmt.Errorf("invalid expiry date")
    }
    
    // Validate CVV
    if !ccp.validator.IsValidCVV(request.PaymentData.CVV, request.PaymentData.CardNumber) {
        return fmt.Errorf("invalid CVV")
    }
    
    // Validate amount
    if request.Amount.LessThanOrEqual(decimal.Zero) {
        return fmt.Errorf("invalid amount: must be positive")
    }
    
    return nil
}

func (ccp *CreditCardProcessor) authenticatePayment(ctx context.Context, request *PaymentRequest) error {
    ccp.logger.Debug("Authenticating credit card payment", zap.String("payment_id", request.PaymentID))
    
    // Perform fraud detection
    riskScore, err := ccp.fraudDetector.CalculateRiskScore(ctx, request)
    if err != nil {
        return fmt.Errorf("fraud detection failed: %w", err)
    }
    
    if riskScore > 0.8 {
        return fmt.Errorf("payment blocked due to high fraud risk: score %.2f", riskScore)
    }
    
    // Tokenize card data
    token, err := ccp.tokenizer.TokenizeCard(request.PaymentData.CardNumber)
    if err != nil {
        return fmt.Errorf("card tokenization failed: %w", err)
    }
    
    // Store token for use in execution
    request.PaymentData.CardToken = token
    
    return nil
}

func (ccp *CreditCardProcessor) executePayment(ctx context.Context, request *PaymentRequest) (*PaymentResponse, error) {
    ccp.logger.Debug("Executing credit card payment", zap.String("payment_id", request.PaymentID))
    
    // Prepare gateway request
    gatewayRequest := &CreditCardGatewayRequest{
        PaymentID:   request.PaymentID,
        Amount:      request.Amount,
        Currency:    request.Currency,
        CardToken:   request.PaymentData.CardToken,
        CustomerID:  request.CustomerID,
        MerchantID:  request.MerchantID,
        Description: request.Description,
    }
    
    // Execute through gateway
    gatewayResponse, err := ccp.gateway.ProcessPayment(ctx, gatewayRequest)
    if err != nil {
        return nil, fmt.Errorf("gateway processing failed: %w", err)
    }
    
    // Convert gateway response to standard response
    response := &PaymentResponse{
        TransactionID: gatewayResponse.TransactionID,
        Status:        gatewayResponse.Status,
        Gateway:       "CREDIT_CARD_GATEWAY",
        Metadata: map[string]interface{}{
            "authorization_code": gatewayResponse.AuthCode,
            "gateway_reference":  gatewayResponse.GatewayRef,
            "card_last_four":     gatewayResponse.CardLastFour,
        },
    }
    
    return response, nil
}

// Override hook methods for credit card specific behavior
func (ccp *CreditCardProcessor) preProcessHook(ctx context.Context, request *PaymentRequest) error {
    // Credit card specific pre-processing
    ccp.logger.Debug("Credit card pre-processing", zap.String("payment_id", request.PaymentID))
    
    // Check if card is blacklisted
    if ccp.validator.IsCardBlacklisted(request.PaymentData.CardNumber) {
        return fmt.Errorf("card is blacklisted")
    }
    
    return nil
}

func (ccp *CreditCardProcessor) onPaymentSuccess(ctx context.Context, result *PaymentResult) error {
    // Call parent success handler
    if err := ccp.BasePaymentProcessor.onPaymentSuccess(ctx, result); err != nil {
        return err
    }
    
    // Credit card specific success handling
    ccp.logger.Info("Credit card payment successful", 
        zap.String("payment_id", result.PaymentID),
        zap.String("transaction_id", result.TransactionID))
    
    // Update card usage statistics
    if err := ccp.updateCardUsageStats(result); err != nil {
        ccp.logger.Warn("Failed to update card usage stats", zap.Error(err))
    }
    
    return nil
}

func (ccp *CreditCardProcessor) updateCardUsageStats(result *PaymentResult) error {
    // Implementation for updating card usage statistics
    return nil
}

// Concrete implementation: Bank Transfer Processor
type BankTransferProcessor struct {
    *BasePaymentProcessor
    gateway         BankTransferGateway
    accountValidator AccountValidator
    limitsChecker   TransferLimitsChecker
}

func NewBankTransferProcessor(base *BasePaymentProcessor, gateway BankTransferGateway) *BankTransferProcessor {
    return &BankTransferProcessor{
        BasePaymentProcessor: base,
        gateway:             gateway,
        accountValidator:    NewAccountValidator(),
        limitsChecker:       NewTransferLimitsChecker(),
    }
}

func (btp *BankTransferProcessor) validatePaymentData(ctx context.Context, request *PaymentRequest) error {
    btp.logger.Debug("Validating bank transfer data", zap.String("payment_id", request.PaymentID))
    
    // Validate account number
    if !btp.accountValidator.IsValidAccountNumber(request.PaymentData.AccountNumber) {
        return fmt.Errorf("invalid account number")
    }
    
    // Validate routing number
    if !btp.accountValidator.IsValidRoutingNumber(request.PaymentData.RoutingNumber) {
        return fmt.Errorf("invalid routing number")
    }
    
    // Check transfer limits
    if err := btp.limitsChecker.CheckTransferLimits(request.CustomerID, request.Amount); err != nil {
        return fmt.Errorf("transfer limits exceeded: %w", err)
    }
    
    return nil
}

func (btp *BankTransferProcessor) authenticatePayment(ctx context.Context, request *PaymentRequest) error {
    btp.logger.Debug("Authenticating bank transfer", zap.String("payment_id", request.PaymentID))
    
    // Verify account ownership
    if err := btp.accountValidator.VerifyAccountOwnership(request.CustomerID, request.PaymentData.AccountNumber); err != nil {
        return fmt.Errorf("account ownership verification failed: %w", err)
    }
    
    // Check account balance (if available)
    if err := btp.accountValidator.CheckSufficientBalance(request.PaymentData.AccountNumber, request.Amount); err != nil {
        return fmt.Errorf("insufficient balance: %w", err)
    }
    
    return nil
}

func (btp *BankTransferProcessor) executePayment(ctx context.Context, request *PaymentRequest) (*PaymentResponse, error) {
    btp.logger.Debug("Executing bank transfer", zap.String("payment_id", request.PaymentID))
    
    // Prepare gateway request
    gatewayRequest := &BankTransferGatewayRequest{
        PaymentID:     request.PaymentID,
        Amount:        request.Amount,
        Currency:      request.Currency,
        AccountNumber: request.PaymentData.AccountNumber,
        RoutingNumber: request.PaymentData.RoutingNumber,
        CustomerID:    request.CustomerID,
        MerchantID:    request.MerchantID,
        Description:   request.Description,
    }
    
    // Execute through gateway
    gatewayResponse, err := btp.gateway.ProcessTransfer(ctx, gatewayRequest)
    if err != nil {
        return nil, fmt.Errorf("gateway processing failed: %w", err)
    }
    
    response := &PaymentResponse{
        TransactionID: gatewayResponse.TransactionID,
        Status:        gatewayResponse.Status,
        Gateway:       "BANK_TRANSFER_GATEWAY",
        Metadata: map[string]interface{}{
            "transfer_reference": gatewayResponse.TransferRef,
            "expected_settlement": gatewayResponse.ExpectedSettlement,
        },
    }
    
    return response, nil
}

func (btp *BankTransferProcessor) postProcessHook(ctx context.Context, result *PaymentResult) error {
    // Bank transfer specific post-processing
    btp.logger.Debug("Bank transfer post-processing", zap.String("payment_id", result.PaymentID))
    
    // Schedule settlement tracking
    if err := btp.scheduleSettlementTracking(result); err != nil {
        btp.logger.Warn("Failed to schedule settlement tracking", zap.Error(err))
    }
    
    return nil
}

func (btp *BankTransferProcessor) scheduleSettlementTracking(result *PaymentResult) error {
    // Implementation for scheduling settlement tracking
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
    PaymentData   PaymentData
    Metadata      map[string]interface{}
}

type PaymentData struct {
    // Credit Card fields
    CardNumber   string
    ExpiryMonth  int
    ExpiryYear   int
    CVV          string
    CardToken    string
    
    // Bank Transfer fields
    AccountNumber string
    RoutingNumber string
}

type PaymentResult struct {
    PaymentID      string
    TransactionID  string
    Status         string
    Amount         decimal.Decimal
    Currency       string
    ProcessedAt    time.Time
    ProcessingTime time.Duration
    Gateway        string
    Metadata       map[string]interface{}
}

type PaymentResponse struct {
    TransactionID string
    Status        string
    Gateway       string
    Metadata      map[string]interface{}
}
```

### 2. Loan Application Processing Template
```go
// Abstract loan application processor
type LoanApplicationProcessor interface {
    ProcessApplication(ctx context.Context, application *LoanApplication) (*ApplicationDecision, error)
    
    // Template method
    processApplicationTemplate(ctx context.Context, application *LoanApplication) (*ApplicationDecision, error)
    
    // Abstract methods
    validateApplication(ctx context.Context, application *LoanApplication) error
    performCreditCheck(ctx context.Context, application *LoanApplication) (*CreditReport, error)
    calculateLoanTerms(ctx context.Context, application *LoanApplication, creditReport *CreditReport) (*LoanTerms, error)
    makeDecision(ctx context.Context, application *LoanApplication, terms *LoanTerms) (*ApplicationDecision, error)
    
    // Hook methods
    preProcessingHook(ctx context.Context, application *LoanApplication) error
    postDecisionHook(ctx context.Context, decision *ApplicationDecision) error
}

// Base loan processor with template method
type BaseLoanProcessor struct {
    logger      *zap.Logger
    metrics     MetricsCollector
    auditor     AuditLogger
    config      LoanConfig
    notifier    NotificationService
}

func (blp *BaseLoanProcessor) ProcessApplication(ctx context.Context, application *LoanApplication) (*ApplicationDecision, error) {
    return blp.processApplicationTemplate(ctx, application)
}

func (blp *BaseLoanProcessor) processApplicationTemplate(ctx context.Context, application *LoanApplication) (*ApplicationDecision, error) {
    startTime := time.Now()
    blp.logger.Info("Starting loan application processing", 
        zap.String("application_id", application.ApplicationID),
        zap.String("loan_type", application.LoanType),
        zap.String("amount", application.RequestedAmount.String()))
    
    // Step 1: Pre-processing hook
    if err := blp.preProcessingHook(ctx, application); err != nil {
        return nil, fmt.Errorf("pre-processing failed: %w", err)
    }
    
    // Step 2: Validate application
    if err := blp.validateApplication(ctx, application); err != nil {
        decision := &ApplicationDecision{
            ApplicationID: application.ApplicationID,
            Status:        "REJECTED",
            Reason:        "Application validation failed: " + err.Error(),
            ProcessedAt:   time.Now(),
        }
        blp.postDecisionHook(ctx, decision)
        return decision, nil
    }
    
    // Step 3: Perform credit check
    creditReport, err := blp.performCreditCheck(ctx, application)
    if err != nil {
        return nil, fmt.Errorf("credit check failed: %w", err)
    }
    
    // Step 4: Calculate loan terms
    loanTerms, err := blp.calculateLoanTerms(ctx, application, creditReport)
    if err != nil {
        return nil, fmt.Errorf("loan terms calculation failed: %w", err)
    }
    
    // Step 5: Make decision
    decision, err := blp.makeDecision(ctx, application, loanTerms)
    if err != nil {
        return nil, fmt.Errorf("decision making failed: %w", err)
    }
    
    decision.ProcessedAt = time.Now()
    decision.ProcessingTime = time.Since(startTime)
    
    // Step 6: Post-decision hook
    if err := blp.postDecisionHook(ctx, decision); err != nil {
        blp.logger.Warn("Post-decision hook failed", zap.Error(err))
    }
    
    // Step 7: Record metrics and audit
    blp.metrics.RecordDuration("loan_processing_duration", time.Since(startTime), application.LoanType)
    blp.auditor.LogApplicationProcessed(ctx, application, decision)
    
    blp.logger.Info("Loan application processing completed", 
        zap.String("application_id", application.ApplicationID),
        zap.String("decision", decision.Status),
        zap.Duration("processing_time", decision.ProcessingTime))
    
    return decision, nil
}

// Default hook implementations
func (blp *BaseLoanProcessor) preProcessingHook(ctx context.Context, application *LoanApplication) error {
    return nil
}

func (blp *BaseLoanProcessor) postDecisionHook(ctx context.Context, decision *ApplicationDecision) error {
    // Send notification to applicant
    if err := blp.notifier.NotifyApplicant(decision); err != nil {
        blp.logger.Warn("Failed to notify applicant", zap.Error(err))
    }
    return nil
}

// Concrete implementation: Personal Loan Processor
type PersonalLoanProcessor struct {
    *BaseLoanProcessor
    creditBureau    CreditBureau
    incomeVerifier  IncomeVerifier
    riskCalculator  RiskCalculator
}

func (plp *PersonalLoanProcessor) validateApplication(ctx context.Context, application *LoanApplication) error {
    // Personal loan specific validation
    if application.ApplicantAge < 18 {
        return fmt.Errorf("applicant must be at least 18 years old")
    }
    
    if application.RequestedAmount.LessThan(decimal.NewFromInt(1000)) {
        return fmt.Errorf("minimum loan amount is $1,000")
    }
    
    if application.RequestedAmount.GreaterThan(decimal.NewFromInt(50000)) {
        return fmt.Errorf("maximum loan amount is $50,000")
    }
    
    return nil
}

func (plp *PersonalLoanProcessor) performCreditCheck(ctx context.Context, application *LoanApplication) (*CreditReport, error) {
    return plp.creditBureau.GetCreditReport(ctx, application.ApplicantSSN)
}

func (plp *PersonalLoanProcessor) calculateLoanTerms(ctx context.Context, application *LoanApplication, creditReport *CreditReport) (*LoanTerms, error) {
    // Personal loan terms calculation
    terms := &LoanTerms{
        PrincipalAmount: application.RequestedAmount,
        Currency:        "USD",
    }
    
    // Calculate interest rate based on credit score
    if creditReport.Score >= 750 {
        terms.InterestRate = decimal.NewFromFloat(5.99)
    } else if creditReport.Score >= 700 {
        terms.InterestRate = decimal.NewFromFloat(7.99)
    } else if creditReport.Score >= 650 {
        terms.InterestRate = decimal.NewFromFloat(12.99)
    } else {
        terms.InterestRate = decimal.NewFromFloat(18.99)
    }
    
    // Set term length
    terms.TermMonths = 36 // 3 years for personal loans
    
    // Calculate monthly payment
    monthlyRate := terms.InterestRate.Div(decimal.NewFromInt(100)).Div(decimal.NewFromInt(12))
    terms.MonthlyPayment = plp.calculateMonthlyPayment(terms.PrincipalAmount, monthlyRate, terms.TermMonths)
    
    return terms, nil
}

func (plp *PersonalLoanProcessor) makeDecision(ctx context.Context, application *LoanApplication, terms *LoanTerms) (*ApplicationDecision, error) {
    decision := &ApplicationDecision{
        ApplicationID: application.ApplicationID,
        LoanTerms:     terms,
    }
    
    // Decision logic for personal loans
    if terms.InterestRate.GreaterThan(decimal.NewFromFloat(15.0)) {
        decision.Status = "REJECTED"
        decision.Reason = "Credit score too low for personal loan"
    } else {
        decision.Status = "APPROVED"
        decision.Reason = "Application meets personal loan criteria"
    }
    
    return decision, nil
}

func (plp *PersonalLoanProcessor) calculateMonthlyPayment(principal, monthlyRate decimal.Decimal, termMonths int) decimal.Decimal {
    // Standard loan payment calculation: P * [r(1+r)^n] / [(1+r)^n - 1]
    onePlusRate := decimal.NewFromInt(1).Add(monthlyRate)
    onePlusRatePowN := onePlusRate.Pow(decimal.NewFromInt(int64(termMonths)))
    
    numerator := principal.Mul(monthlyRate).Mul(onePlusRatePowN)
    denominator := onePlusRatePowN.Sub(decimal.NewFromInt(1))
    
    return numerator.Div(denominator)
}

// Concrete implementation: Mortgage Processor
type MortgageProcessor struct {
    *BaseLoanProcessor
    propertyAppraiser PropertyAppraiser
    titleService      TitleService
    insuranceValidator InsuranceValidator
}

func (mp *MortgageProcessor) validateApplication(ctx context.Context, application *LoanApplication) error {
    // Mortgage specific validation
    if application.ApplicantAge < 21 {
        return fmt.Errorf("applicant must be at least 21 years old for mortgage")
    }
    
    if application.RequestedAmount.LessThan(decimal.NewFromInt(50000)) {
        return fmt.Errorf("minimum mortgage amount is $50,000")
    }
    
    if application.PropertyAddress == "" {
        return fmt.Errorf("property address is required for mortgage")
    }
    
    return nil
}

func (mp *MortgageProcessor) performCreditCheck(ctx context.Context, application *LoanApplication) (*CreditReport, error) {
    // Enhanced credit check for mortgage
    creditReport, err := mp.creditBureau.GetCreditReport(ctx, application.ApplicantSSN)
    if err != nil {
        return nil, err
    }
    
    // Additional checks for mortgage
    if creditReport.Score < 620 {
        return nil, fmt.Errorf("credit score too low for mortgage: %d", creditReport.Score)
    }
    
    return creditReport, nil
}

func (mp *MortgageProcessor) calculateLoanTerms(ctx context.Context, application *LoanApplication, creditReport *CreditReport) (*LoanTerms, error) {
    // Property appraisal
    appraisal, err := mp.propertyAppraiser.AppraiseProperty(ctx, application.PropertyAddress)
    if err != nil {
        return nil, fmt.Errorf("property appraisal failed: %w", err)
    }
    
    // Loan-to-value ratio check
    ltvRatio := application.RequestedAmount.Div(appraisal.Value)
    if ltvRatio.GreaterThan(decimal.NewFromFloat(0.95)) {
        return nil, fmt.Errorf("loan-to-value ratio too high: %.2f%%", ltvRatio.InexactFloat64()*100)
    }
    
    terms := &LoanTerms{
        PrincipalAmount: application.RequestedAmount,
        Currency:        "USD",
        TermMonths:      360, // 30 years for mortgage
        PropertyValue:   appraisal.Value,
        LTVRatio:        ltvRatio,
    }
    
    // Calculate mortgage interest rate
    baseRate := decimal.NewFromFloat(3.5)
    if creditReport.Score >= 760 {
        terms.InterestRate = baseRate
    } else if creditReport.Score >= 720 {
        terms.InterestRate = baseRate.Add(decimal.NewFromFloat(0.25))
    } else if creditReport.Score >= 680 {
        terms.InterestRate = baseRate.Add(decimal.NewFromFloat(0.5))
    } else {
        terms.InterestRate = baseRate.Add(decimal.NewFromFloat(1.0))
    }
    
    // Calculate monthly payment
    monthlyRate := terms.InterestRate.Div(decimal.NewFromInt(100)).Div(decimal.NewFromInt(12))
    terms.MonthlyPayment = mp.calculateMonthlyPayment(terms.PrincipalAmount, monthlyRate, terms.TermMonths)
    
    return terms, nil
}

func (mp *MortgageProcessor) makeDecision(ctx context.Context, application *LoanApplication, terms *LoanTerms) (*ApplicationDecision, error) {
    decision := &ApplicationDecision{
        ApplicationID: application.ApplicationID,
        LoanTerms:     terms,
    }
    
    // Debt-to-income ratio check
    monthlyIncome := application.MonthlyIncome
    totalMonthlyDebt := application.MonthlyDebtPayments.Add(terms.MonthlyPayment)
    dtiRatio := totalMonthlyDebt.Div(monthlyIncome)
    
    if dtiRatio.GreaterThan(decimal.NewFromFloat(0.43)) {
        decision.Status = "REJECTED"
        decision.Reason = fmt.Sprintf("Debt-to-income ratio too high: %.2f%%", dtiRatio.InexactFloat64()*100)
    } else {
        decision.Status = "APPROVED"
        decision.Reason = "Application meets mortgage criteria"
    }
    
    return decision, nil
}

func (mp *MortgageProcessor) calculateMonthlyPayment(principal, monthlyRate decimal.Decimal, termMonths int) decimal.Decimal {
    // Same calculation as personal loan
    onePlusRate := decimal.NewFromInt(1).Add(monthlyRate)
    onePlusRatePowN := onePlusRate.Pow(decimal.NewFromInt(int64(termMonths)))
    
    numerator := principal.Mul(monthlyRate).Mul(onePlusRatePowN)
    denominator := onePlusRatePowN.Sub(decimal.NewFromInt(1))
    
    return numerator.Div(denominator)
}

func (mp *MortgageProcessor) preProcessingHook(ctx context.Context, application *LoanApplication) error {
    // Mortgage specific pre-processing
    mp.logger.Debug("Mortgage pre-processing", zap.String("application_id", application.ApplicationID))
    
    // Verify property title
    if err := mp.titleService.VerifyTitle(ctx, application.PropertyAddress); err != nil {
        return fmt.Errorf("title verification failed: %w", err)
    }
    
    return nil
}

func (mp *MortgageProcessor) postDecisionHook(ctx context.Context, decision *ApplicationDecision) error {
    // Call parent post-decision hook
    if err := mp.BaseLoanProcessor.postDecisionHook(ctx, decision); err != nil {
        return err
    }
    
    // Mortgage specific post-decision processing
    if decision.Status == "APPROVED" {
        // Schedule closing process
        if err := mp.scheduleClosing(decision); err != nil {
            mp.logger.Warn("Failed to schedule closing", zap.Error(err))
        }
    }
    
    return nil
}

func (mp *MortgageProcessor) scheduleClosing(decision *ApplicationDecision) error {
    // Implementation for scheduling closing process
    return nil
}
```

### 3. KYC Verification Template
```go
// Abstract KYC processor template
type KYCProcessor interface {
    ProcessKYC(ctx context.Context, customer *Customer) (*KYCResult, error)
    
    // Template method
    processKYCTemplate(ctx context.Context, customer *Customer) (*KYCResult, error)
    
    // Abstract methods
    validateDocuments(ctx context.Context, customer *Customer) error
    verifyIdentity(ctx context.Context, customer *Customer) (*IdentityVerification, error)
    performRiskAssessment(ctx context.Context, customer *Customer) (*RiskAssessment, error)
    makeKYCDecision(ctx context.Context, customer *Customer, verification *IdentityVerification, risk *RiskAssessment) (*KYCDecision, error)
    
    // Hook methods
    preKYCHook(ctx context.Context, customer *Customer) error
    postKYCHook(ctx context.Context, result *KYCResult) error
}

// Base KYC processor
type BaseKYCProcessor struct {
    logger          *zap.Logger
    documentService DocumentService
    complianceDB    ComplianceDatabase
    auditor         AuditLogger
    config          KYCConfig
}

func (bkp *BaseKYCProcessor) ProcessKYC(ctx context.Context, customer *Customer) (*KYCResult, error) {
    return bkp.processKYCTemplate(ctx, customer)
}

func (bkp *BaseKYCProcessor) processKYCTemplate(ctx context.Context, customer *Customer) (*KYCResult, error) {
    startTime := time.Now()
    bkp.logger.Info("Starting KYC processing", 
        zap.String("customer_id", customer.CustomerID),
        zap.String("customer_type", customer.Type))
    
    // Step 1: Pre-KYC hook
    if err := bkp.preKYCHook(ctx, customer); err != nil {
        return nil, fmt.Errorf("pre-KYC processing failed: %w", err)
    }
    
    // Step 2: Validate documents
    if err := bkp.validateDocuments(ctx, customer); err != nil {
        result := &KYCResult{
            CustomerID:  customer.CustomerID,
            Status:      "REJECTED",
            Reason:      "Document validation failed: " + err.Error(),
            ProcessedAt: time.Now(),
        }
        bkp.postKYCHook(ctx, result)
        return result, nil
    }
    
    // Step 3: Verify identity
    identityVerification, err := bkp.verifyIdentity(ctx, customer)
    if err != nil {
        return nil, fmt.Errorf("identity verification failed: %w", err)
    }
    
    // Step 4: Perform risk assessment
    riskAssessment, err := bkp.performRiskAssessment(ctx, customer)
    if err != nil {
        return nil, fmt.Errorf("risk assessment failed: %w", err)
    }
    
    // Step 5: Make KYC decision
    decision, err := bkp.makeKYCDecision(ctx, customer, identityVerification, riskAssessment)
    if err != nil {
        return nil, fmt.Errorf("KYC decision making failed: %w", err)
    }
    
    // Step 6: Create result
    result := &KYCResult{
        CustomerID:           customer.CustomerID,
        Status:               decision.Status,
        Reason:               decision.Reason,
        RiskLevel:           riskAssessment.RiskLevel,
        IdentityScore:       identityVerification.Score,
        RequiredDocuments:   decision.RequiredDocuments,
        ProcessedAt:         time.Now(),
        ProcessingTime:      time.Since(startTime),
        ExpiryDate:          decision.ExpiryDate,
    }
    
    // Step 7: Post-KYC hook
    if err := bkp.postKYCHook(ctx, result); err != nil {
        bkp.logger.Warn("Post-KYC hook failed", zap.Error(err))
    }
    
    // Step 8: Audit and compliance logging
    bkp.auditor.LogKYCProcessed(ctx, customer, result)
    bkp.complianceDB.RecordKYCEvent(ctx, result)
    
    bkp.logger.Info("KYC processing completed", 
        zap.String("customer_id", customer.CustomerID),
        zap.String("status", result.Status),
        zap.Duration("processing_time", result.ProcessingTime))
    
    return result, nil
}

// Default hook implementations
func (bkp *BaseKYCProcessor) preKYCHook(ctx context.Context, customer *Customer) error {
    return nil
}

func (bkp *BaseKYCProcessor) postKYCHook(ctx context.Context, result *KYCResult) error {
    // Default: record in compliance database
    return bkp.complianceDB.RecordKYCResult(ctx, result)
}

// Concrete implementation: Individual KYC Processor
type IndividualKYCProcessor struct {
    *BaseKYCProcessor
    identityService    IdentityService
    sanctionsChecker   SanctionsChecker
    pepChecker         PEPChecker
    documentValidator  DocumentValidator
}

func (ikp *IndividualKYCProcessor) validateDocuments(ctx context.Context, customer *Customer) error {
    // Individual-specific document validation
    requiredDocs := []string{"GOVERNMENT_ID", "PROOF_OF_ADDRESS"}
    
    for _, docType := range requiredDocs {
        if !ikp.hasDocument(customer, docType) {
            return fmt.Errorf("missing required document: %s", docType)
        }
        
        doc := ikp.getDocument(customer, docType)
        if err := ikp.documentValidator.ValidateDocument(doc); err != nil {
            return fmt.Errorf("invalid %s: %w", docType, err)
        }
    }
    
    return nil
}

func (ikp *IndividualKYCProcessor) verifyIdentity(ctx context.Context, customer *Customer) (*IdentityVerification, error) {
    govID := ikp.getDocument(customer, "GOVERNMENT_ID")
    
    verification, err := ikp.identityService.VerifyGovernmentID(ctx, govID)
    if err != nil {
        return nil, err
    }
    
    // Face matching if photo available
    if customer.PhotoURL != "" {
        faceMatch, err := ikp.identityService.VerifyFaceMatch(ctx, customer.PhotoURL, govID.PhotoURL)
        if err != nil {
            ikp.logger.Warn("Face matching failed", zap.Error(err))
        } else {
            verification.FaceMatchScore = faceMatch.Score
        }
    }
    
    return verification, nil
}

func (ikp *IndividualKYCProcessor) performRiskAssessment(ctx context.Context, customer *Customer) (*RiskAssessment, error) {
    assessment := &RiskAssessment{
        CustomerID: customer.CustomerID,
        RiskLevel:  "LOW",
        Score:      0.0,
        Factors:    make([]string, 0),
    }
    
    // Check sanctions lists
    sanctionsResult, err := ikp.sanctionsChecker.CheckSanctions(ctx, customer)
    if err != nil {
        return nil, err
    }
    
    if sanctionsResult.IsOnSanctionsList {
        assessment.RiskLevel = "HIGH"
        assessment.Score += 100.0
        assessment.Factors = append(assessment.Factors, "SANCTIONS_LIST_MATCH")
    }
    
    // Check PEP lists
    pepResult, err := ikp.pepChecker.CheckPEP(ctx, customer)
    if err != nil {
        return nil, err
    }
    
    if pepResult.IsPEP {
        assessment.RiskLevel = "HIGH"
        assessment.Score += 50.0
        assessment.Factors = append(assessment.Factors, "POLITICALLY_EXPOSED_PERSON")
    }
    
    // Geographic risk assessment
    if ikp.isHighRiskCountry(customer.Country) {
        assessment.Score += 30.0
        assessment.Factors = append(assessment.Factors, "HIGH_RISK_JURISDICTION")
        if assessment.RiskLevel == "LOW" {
            assessment.RiskLevel = "MEDIUM"
        }
    }
    
    return assessment, nil
}

func (ikp *IndividualKYCProcessor) makeKYCDecision(ctx context.Context, customer *Customer, verification *IdentityVerification, risk *RiskAssessment) (*KYCDecision, error) {
    decision := &KYCDecision{
        CustomerID: customer.CustomerID,
        ExpiryDate: time.Now().AddDate(1, 0, 0), // 1 year expiry
    }
    
    // Decision logic
    if risk.RiskLevel == "HIGH" {
        decision.Status = "REJECTED"
        decision.Reason = "High risk customer - manual review required"
        decision.RequiredDocuments = []string{"ENHANCED_DUE_DILIGENCE"}
    } else if verification.Score < 0.8 {
        decision.Status = "PENDING"
        decision.Reason = "Identity verification score too low"
        decision.RequiredDocuments = []string{"ADDITIONAL_ID_VERIFICATION"}
    } else {
        decision.Status = "APPROVED"
        decision.Reason = "KYC verification successful"
    }
    
    return decision, nil
}

func (ikp *IndividualKYCProcessor) hasDocument(customer *Customer, docType string) bool {
    for _, doc := range customer.Documents {
        if doc.Type == docType {
            return true
        }
    }
    return false
}

func (ikp *IndividualKYCProcessor) getDocument(customer *Customer, docType string) *Document {
    for _, doc := range customer.Documents {
        if doc.Type == docType {
            return doc
        }
    }
    return nil
}

func (ikp *IndividualKYCProcessor) isHighRiskCountry(country string) bool {
    highRiskCountries := []string{"NK", "IR", "SY", "AF"} // Example list
    for _, c := range highRiskCountries {
        if c == country {
            return true
        }
    }
    return false
}

// Supporting types
type Customer struct {
    CustomerID string
    Type       string // "INDIVIDUAL", "BUSINESS"
    FirstName  string
    LastName   string
    Country    string
    PhotoURL   string
    Documents  []*Document
}

type Document struct {
    Type     string
    URL      string
    PhotoURL string
    Data     map[string]interface{}
}

type KYCResult struct {
    CustomerID        string
    Status            string
    Reason            string
    RiskLevel         string
    IdentityScore     float64
    RequiredDocuments []string
    ProcessedAt       time.Time
    ProcessingTime    time.Duration
    ExpiryDate        time.Time
}

type IdentityVerification struct {
    Score          float64
    FaceMatchScore float64
    IsValid        bool
    Confidence     float64
}

type RiskAssessment struct {
    CustomerID string
    RiskLevel  string
    Score      float64
    Factors    []string
}

type KYCDecision struct {
    CustomerID        string
    Status            string
    Reason            string
    RequiredDocuments []string
    ExpiryDate        time.Time
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

// Example: Document generation system using Template Method
// This demonstrates how different document types follow the same generation process
// but with different specific implementations for each step

// Abstract document generator
type DocumentGenerator interface {
    GenerateDocument() (*Document, error)
    
    // Template method
    generateDocumentTemplate() (*Document, error)
    
    // Abstract methods that must be implemented
    createHeader() (*Header, error)
    createBody() (*Body, error)
    createFooter() (*Footer, error)
    formatDocument(*Document) error
    
    // Hook methods that can be overridden
    preGenerationHook() error
    postGenerationHook(*Document) error
    validateContent(*Document) error
}

// Base document generator with template method implementation
type BaseDocumentGenerator struct {
    title       string
    author      string
    createdAt   time.Time
    logger      *zap.Logger
}

func NewBaseDocumentGenerator(title, author string, logger *zap.Logger) *BaseDocumentGenerator {
    return &BaseDocumentGenerator{
        title:     title,
        author:    author,
        createdAt: time.Now(),
        logger:    logger,
    }
}

// Template method - defines the document generation algorithm
func (bdg *BaseDocumentGenerator) GenerateDocument() (*Document, error) {
    return bdg.generateDocumentTemplate()
}

func (bdg *BaseDocumentGenerator) generateDocumentTemplate() (*Document, error) {
    bdg.logger.Info("Starting document generation", 
        zap.String("title", bdg.title),
        zap.String("author", bdg.author))
    
    // Step 1: Pre-generation hook
    if err := bdg.preGenerationHook(); err != nil {
        return nil, fmt.Errorf("pre-generation hook failed: %w", err)
    }
    
    // Step 2: Create header
    header, err := bdg.createHeader()
    if err != nil {
        return nil, fmt.Errorf("header creation failed: %w", err)
    }
    
    // Step 3: Create body
    body, err := bdg.createBody()
    if err != nil {
        return nil, fmt.Errorf("body creation failed: %w", err)
    }
    
    // Step 4: Create footer
    footer, err := bdg.createFooter()
    if err != nil {
        return nil, fmt.Errorf("footer creation failed: %w", err)
    }
    
    // Step 5: Assemble document
    document := &Document{
        Header:      header,
        Body:        body,
        Footer:      footer,
        Title:       bdg.title,
        Author:      bdg.author,
        CreatedAt:   bdg.createdAt,
        GeneratedAt: time.Now(),
    }
    
    // Step 6: Format document
    if err := bdg.formatDocument(document); err != nil {
        return nil, fmt.Errorf("document formatting failed: %w", err)
    }
    
    // Step 7: Validate content
    if err := bdg.validateContent(document); err != nil {
        return nil, fmt.Errorf("content validation failed: %w", err)
    }
    
    // Step 8: Post-generation hook
    if err := bdg.postGenerationHook(document); err != nil {
        bdg.logger.Warn("Post-generation hook failed", zap.Error(err))
        // Don't fail document generation for post-processing errors
    }
    
    bdg.logger.Info("Document generation completed", 
        zap.String("title", document.Title),
        zap.Duration("generation_time", time.Since(bdg.createdAt)))
    
    return document, nil
}

// Default hook implementations
func (bdg *BaseDocumentGenerator) preGenerationHook() error {
    bdg.logger.Debug("Pre-generation hook executed")
    return nil
}

func (bdg *BaseDocumentGenerator) postGenerationHook(document *Document) error {
    bdg.logger.Debug("Post-generation hook executed", 
        zap.String("document_title", document.Title))
    return nil
}

func (bdg *BaseDocumentGenerator) validateContent(document *Document) error {
    // Basic validation
    if document.Header == nil {
        return fmt.Errorf("document header is missing")
    }
    if document.Body == nil {
        return fmt.Errorf("document body is missing")
    }
    if document.Footer == nil {
        return fmt.Errorf("document footer is missing")
    }
    return nil
}

// Concrete implementation: Invoice Generator
type InvoiceGenerator struct {
    *BaseDocumentGenerator
    invoiceNumber string
    customerInfo  *CustomerInfo
    items         []*InvoiceItem
    taxRate       float64
}

func NewInvoiceGenerator(invoiceNumber string, customer *CustomerInfo, items []*InvoiceItem, taxRate float64, logger *zap.Logger) *InvoiceGenerator {
    base := NewBaseDocumentGenerator(fmt.Sprintf("Invoice %s", invoiceNumber), "Invoice System", logger)
    
    return &InvoiceGenerator{
        BaseDocumentGenerator: base,
        invoiceNumber:        invoiceNumber,
        customerInfo:         customer,
        items:               items,
        taxRate:             taxRate,
    }
}

// Implement abstract methods for invoice generation
func (ig *InvoiceGenerator) createHeader() (*Header, error) {
    ig.logger.Debug("Creating invoice header", zap.String("invoice_number", ig.invoiceNumber))
    
    header := &Header{
        Type: "INVOICE",
        Content: map[string]interface{}{
            "invoice_number": ig.invoiceNumber,
            "issue_date":     time.Now().Format("2006-01-02"),
            "due_date":       time.Now().AddDate(0, 0, 30).Format("2006-01-02"),
            "company_name":   "Acme Corporation",
            "company_address": "123 Business St, City, State 12345",
            "company_phone":  "+1-555-0123",
            "company_email":  "billing@acme.com",
        },
    }
    
    return header, nil
}

func (ig *InvoiceGenerator) createBody() (*Body, error) {
    ig.logger.Debug("Creating invoice body", 
        zap.String("customer_name", ig.customerInfo.Name),
        zap.Int("item_count", len(ig.items)))
    
    // Calculate totals
    subtotal := 0.0
    for _, item := range ig.items {
        subtotal += item.Quantity * item.UnitPrice
    }
    
    tax := subtotal * ig.taxRate
    total := subtotal + tax
    
    body := &Body{
        Type: "INVOICE_BODY",
        Content: map[string]interface{}{
            "customer_info": map[string]interface{}{
                "name":    ig.customerInfo.Name,
                "address": ig.customerInfo.Address,
                "phone":   ig.customerInfo.Phone,
                "email":   ig.customerInfo.Email,
            },
            "items":    ig.items,
            "subtotal": subtotal,
            "tax_rate": ig.taxRate,
            "tax":      tax,
            "total":    total,
        },
    }
    
    return body, nil
}

func (ig *InvoiceGenerator) createFooter() (*Footer, error) {
    ig.logger.Debug("Creating invoice footer")
    
    footer := &Footer{
        Type: "INVOICE_FOOTER",
        Content: map[string]interface{}{
            "payment_terms": "Payment due within 30 days",
            "payment_methods": []string{"Credit Card", "Bank Transfer", "Check"},
            "contact_info": "For questions, contact billing@acme.com",
            "legal_notice": "This invoice is generated electronically and is valid without signature",
        },
    }
    
    return footer, nil
}

func (ig *InvoiceGenerator) formatDocument(document *Document) error {
    ig.logger.Debug("Formatting invoice document")
    
    // Invoice-specific formatting
    document.Format = "PDF"
    document.Layout = "INVOICE_LAYOUT"
    document.Styling = map[string]interface{}{
        "font_family": "Arial",
        "font_size":   12,
        "colors": map[string]string{
            "header":  "#2C3E50",
            "primary": "#3498DB",
            "text":    "#2C3E50",
        },
        "logo_position": "top-left",
        "table_style":   "striped",
    }
    
    return nil
}

// Override hook methods for invoice-specific behavior
func (ig *InvoiceGenerator) preGenerationHook() error {
    ig.logger.Debug("Invoice pre-generation hook")
    
    // Validate invoice data
    if ig.invoiceNumber == "" {
        return fmt.Errorf("invoice number is required")
    }
    
    if ig.customerInfo == nil {
        return fmt.Errorf("customer information is required")
    }
    
    if len(ig.items) == 0 {
        return fmt.Errorf("at least one invoice item is required")
    }
    
    return nil
}

func (ig *InvoiceGenerator) postGenerationHook(document *Document) error {
    // Call parent hook
    if err := ig.BaseDocumentGenerator.postGenerationHook(document); err != nil {
        return err
    }
    
    // Invoice-specific post-processing
    ig.logger.Info("Invoice generated successfully", 
        zap.String("invoice_number", ig.invoiceNumber),
        zap.String("customer", ig.customerInfo.Name))
    
    // Save to invoice database
    if err := ig.saveInvoiceToDatabase(document); err != nil {
        ig.logger.Warn("Failed to save invoice to database", zap.Error(err))
    }
    
    // Send email notification
    if err := ig.sendInvoiceNotification(document); err != nil {
        ig.logger.Warn("Failed to send invoice notification", zap.Error(err))
    }
    
    return nil
}

func (ig *InvoiceGenerator) saveInvoiceToDatabase(document *Document) error {
    // Simulate database save
    ig.logger.Debug("Saving invoice to database", 
        zap.String("invoice_number", ig.invoiceNumber))
    return nil
}

func (ig *InvoiceGenerator) sendInvoiceNotification(document *Document) error {
    // Simulate email notification
    ig.logger.Debug("Sending invoice notification", 
        zap.String("customer_email", ig.customerInfo.Email))
    return nil
}

// Concrete implementation: Report Generator
type ReportGenerator struct {
    *BaseDocumentGenerator
    reportType string
    dateRange  DateRange
    data       interface{}
    charts     []*Chart
}

func NewReportGenerator(reportType string, dateRange DateRange, data interface{}, charts []*Chart, logger *zap.Logger) *ReportGenerator {
    base := NewBaseDocumentGenerator(fmt.Sprintf("%s Report", reportType), "Report System", logger)
    
    return &ReportGenerator{
        BaseDocumentGenerator: base,
        reportType:           reportType,
        dateRange:            dateRange,
        data:                data,
        charts:              charts,
    }
}

func (rg *ReportGenerator) createHeader() (*Header, error) {
    rg.logger.Debug("Creating report header", zap.String("report_type", rg.reportType))
    
    header := &Header{
        Type: "REPORT",
        Content: map[string]interface{}{
            "report_type":   rg.reportType,
            "date_range":    fmt.Sprintf("%s to %s", rg.dateRange.Start.Format("2006-01-02"), rg.dateRange.End.Format("2006-01-02")),
            "generated_at":  time.Now().Format("2006-01-02 15:04:05"),
            "company_name":  "Analytics Corp",
            "logo_url":      "/assets/logo.png",
        },
    }
    
    return header, nil
}

func (rg *ReportGenerator) createBody() (*Body, error) {
    rg.logger.Debug("Creating report body", 
        zap.String("report_type", rg.reportType),
        zap.Int("chart_count", len(rg.charts)))
    
    body := &Body{
        Type: "REPORT_BODY",
        Content: map[string]interface{}{
            "summary":    rg.generateSummary(),
            "data":       rg.data,
            "charts":     rg.charts,
            "insights":   rg.generateInsights(),
            "date_range": rg.dateRange,
        },
    }
    
    return body, nil
}

func (rg *ReportGenerator) createFooter() (*Footer, error) {
    rg.logger.Debug("Creating report footer")
    
    footer := &Footer{
        Type: "REPORT_FOOTER",
        Content: map[string]interface{}{
            "disclaimer": "This report is generated automatically and may contain estimated values",
            "contact": "For questions about this report, contact analytics@company.com",
            "page_numbers": true,
            "confidentiality": "CONFIDENTIAL - Internal Use Only",
        },
    }
    
    return footer, nil
}

func (rg *ReportGenerator) formatDocument(document *Document) error {
    rg.logger.Debug("Formatting report document")
    
    // Report-specific formatting
    document.Format = "PDF"
    document.Layout = "REPORT_LAYOUT"
    document.Styling = map[string]interface{}{
        "font_family": "Helvetica",
        "font_size":   11,
        "colors": map[string]string{
            "header":    "#1A237E",
            "primary":   "#3F51B5",
            "secondary": "#9C27B0",
            "text":      "#212121",
        },
        "chart_style": "professional",
        "margins": map[string]int{
            "top":    72,
            "bottom": 72,
            "left":   72,
            "right":  72,
        },
    }
    
    return nil
}

func (rg *ReportGenerator) generateSummary() map[string]interface{} {
    return map[string]interface{}{
        "total_records": 1000,
        "time_period":   fmt.Sprintf("%d days", rg.dateRange.End.Sub(rg.dateRange.Start).Hours()/24),
        "key_metrics": map[string]float64{
            "growth_rate":   5.2,
            "conversion":    12.8,
            "satisfaction":  4.3,
        },
    }
}

func (rg *ReportGenerator) generateInsights() []string {
    return []string{
        "Performance showed steady improvement over the reporting period",
        "Conversion rates exceeded targets by 15%",
        "Customer satisfaction remains high at 4.3/5.0",
    }
}

func (rg *ReportGenerator) postGenerationHook(document *Document) error {
    // Call parent hook
    if err := rg.BaseDocumentGenerator.postGenerationHook(document); err != nil {
        return err
    }
    
    // Report-specific post-processing
    rg.logger.Info("Report generated successfully", 
        zap.String("report_type", rg.reportType),
        zap.String("date_range", fmt.Sprintf("%s to %s", 
            rg.dateRange.Start.Format("2006-01-02"), 
            rg.dateRange.End.Format("2006-01-02"))))
    
    // Archive report
    if err := rg.archiveReport(document); err != nil {
        rg.logger.Warn("Failed to archive report", zap.Error(err))
    }
    
    return nil
}

func (rg *ReportGenerator) archiveReport(document *Document) error {
    // Simulate report archiving
    rg.logger.Debug("Archiving report", zap.String("report_type", rg.reportType))
    return nil
}

// Supporting types
type Document struct {
    Header      *Header
    Body        *Body
    Footer      *Footer
    Title       string
    Author      string
    CreatedAt   time.Time
    GeneratedAt time.Time
    Format      string
    Layout      string
    Styling     map[string]interface{}
}

type Header struct {
    Type    string
    Content map[string]interface{}
}

type Body struct {
    Type    string
    Content map[string]interface{}
}

type Footer struct {
    Type    string
    Content map[string]interface{}
}

type CustomerInfo struct {
    Name    string
    Address string
    Phone   string
    Email   string
}

type InvoiceItem struct {
    Description string
    Quantity    float64
    UnitPrice   float64
}

type DateRange struct {
    Start time.Time
    End   time.Time
}

type Chart struct {
    Type   string
    Title  string
    Data   interface{}
    Config map[string]interface{}
}

// Example usage
func main() {
    fmt.Println("=== Template Method Pattern Demo ===\n")
    
    // Create logger
    logger, _ := zap.NewDevelopment()
    defer logger.Sync()
    
    // Example 1: Generate Invoice
    fmt.Println("=== Generating Invoice ===")
    
    customer := &CustomerInfo{
        Name:    "John Doe",
        Address: "456 Customer Lane, City, State 67890",
        Phone:   "+1-555-0199",
        Email:   "john.doe@email.com",
    }
    
    items := []*InvoiceItem{
        {Description: "Software License", Quantity: 1, UnitPrice: 299.99},
        {Description: "Support Package", Quantity: 1, UnitPrice: 99.99},
        {Description: "Training Hours", Quantity: 4, UnitPrice: 150.00},
    }
    
    invoiceGen := NewInvoiceGenerator("INV-2024-001", customer, items, 0.08, logger)
    
    invoice, err := invoiceGen.GenerateDocument()
    if err != nil {
        fmt.Printf("Invoice generation failed: %v\n", err)
    } else {
        fmt.Printf("Invoice generated successfully!\n")
        fmt.Printf("  Title: %s\n", invoice.Title)
        fmt.Printf("  Author: %s\n", invoice.Author)
        fmt.Printf("  Format: %s\n", invoice.Format)
        fmt.Printf("  Layout: %s\n", invoice.Layout)
        fmt.Printf("  Generated at: %s\n", invoice.GeneratedAt.Format("2006-01-02 15:04:05"))
        
        // Display invoice details
        if headerContent, ok := invoice.Header.Content["invoice_number"]; ok {
            fmt.Printf("  Invoice Number: %s\n", headerContent)
        }
        
        if bodyContent, ok := invoice.Body.Content["total"]; ok {
            fmt.Printf("  Total Amount: $%.2f\n", bodyContent)
        }
    }
    
    fmt.Println()
    
    // Example 2: Generate Report
    fmt.Println("=== Generating Report ===")
    
    dateRange := DateRange{
        Start: time.Now().AddDate(0, -1, 0), // 1 month ago
        End:   time.Now(),
    }
    
    charts := []*Chart{
        {
            Type:  "line",
            Title: "Revenue Trend",
            Data:  []float64{1000, 1200, 1100, 1400, 1300},
            Config: map[string]interface{}{
                "color": "#3498DB",
                "line_width": 2,
            },
        },
        {
            Type:  "bar",
            Title: "Sales by Region",
            Data: map[string]float64{
                "North": 2500,
                "South": 1800,
                "East":  2200,
                "West":  1900,
            },
            Config: map[string]interface{}{
                "colors": []string{"#E74C3C", "#F39C12", "#2ECC71", "#9B59B6"},
            },
        },
    }
    
    reportData := map[string]interface{}{
        "sales": map[string]float64{
            "total":     8400,
            "growth":    5.2,
            "target":    8000,
        },
        "customers": map[string]int{
            "new":       45,
            "returning": 234,
            "churned":   12,
        },
    }
    
    reportGen := NewReportGenerator("Sales Performance", dateRange, reportData, charts, logger)
    
    report, err := reportGen.GenerateDocument()
    if err != nil {
        fmt.Printf("Report generation failed: %v\n", err)
    } else {
        fmt.Printf("Report generated successfully!\n")
        fmt.Printf("  Title: %s\n", report.Title)
        fmt.Printf("  Author: %s\n", report.Author)
        fmt.Printf("  Format: %s\n", report.Format)
        fmt.Printf("  Layout: %s\n", report.Layout)
        fmt.Printf("  Generated at: %s\n", report.GeneratedAt.Format("2006-01-02 15:04:05"))
        
        // Display report details
        if headerContent, ok := report.Header.Content["report_type"]; ok {
            fmt.Printf("  Report Type: %s\n", headerContent)
        }
        
        if headerContent, ok := report.Header.Content["date_range"]; ok {
            fmt.Printf("  Date Range: %s\n", headerContent)
        }
        
        if bodyContent, ok := report.Body.Content["summary"]; ok {
            if summary, ok := bodyContent.(map[string]interface{}); ok {
                if totalRecords, ok := summary["total_records"]; ok {
                    fmt.Printf("  Total Records: %v\n", totalRecords)
                }
            }
        }
    }
    
    fmt.Println()
    
    // Example 3: Demonstrate Template Method Structure
    fmt.Println("=== Template Method Structure Demo ===")
    fmt.Println("Both generators follow the same algorithm:")
    fmt.Println("1. Pre-generation hook")
    fmt.Println("2. Create header")
    fmt.Println("3. Create body")
    fmt.Println("4. Create footer")
    fmt.Println("5. Format document")
    fmt.Println("6. Validate content")
    fmt.Println("7. Post-generation hook")
    fmt.Println()
    fmt.Println("But each implements the steps differently:")
    fmt.Println("- Invoice: Business-focused formatting, tax calculations")
    fmt.Println("- Report: Analytics-focused formatting, charts and insights")
    fmt.Println()
    
    fmt.Println("=== Template Method Pattern Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Interface-based Template Method**
```go
// Using interfaces instead of inheritance
type DocumentProcessor interface {
    ProcessDocument() error
}

type DocumentStep interface {
    Execute(context *ProcessingContext) error
}

type TemplateDocumentProcessor struct {
    steps []DocumentStep
}

func (tdp *TemplateDocumentProcessor) ProcessDocument() error {
    context := &ProcessingContext{}
    
    for _, step := range tdp.steps {
        if err := step.Execute(context); err != nil {
            return err
        }
    }
    
    return nil
}
```

2. **Functional Template Method**
```go
// Using function composition
type ProcessingStep func(*ProcessingContext) error

type FunctionalProcessor struct {
    steps []ProcessingStep
}

func (fp *FunctionalProcessor) Process() error {
    context := &ProcessingContext{}
    
    for _, step := range fp.steps {
        if err := step(context); err != nil {
            return err
        }
    }
    
    return nil
}

// Build processors with function chains
func NewPaymentProcessor() *FunctionalProcessor {
    return &FunctionalProcessor{
        steps: []ProcessingStep{
            validatePaymentStep,
            authenticateStep,
            processPaymentStep,
            recordMetricsStep,
        },
    }
}
```

3. **Pipeline Template Method**
```go
type Pipeline struct {
    stages []Stage
}

type Stage interface {
    Process(input interface{}) (interface{}, error)
    CanProcess(input interface{}) bool
}

func (p *Pipeline) Execute(input interface{}) (interface{}, error) {
    current := input
    
    for _, stage := range p.stages {
        if !stage.CanProcess(current) {
            continue
        }
        
        result, err := stage.Process(current)
        if err != nil {
            return nil, err
        }
        current = result
    }
    
    return current, nil
}
```

### Trade-offs

**Pros:**
- **Code Reuse**: Common algorithm structure shared across implementations
- **Consistency**: Ensures all implementations follow the same process
- **Maintainability**: Changes to algorithm structure in one place
- **Framework Development**: Great for creating extensible frameworks
- **Inversion of Control**: Framework controls the algorithm flow

**Cons:**
- **Inheritance Coupling**: Creates dependency on base class
- **Rigid Structure**: Difficult to change algorithm structure at runtime
- **Complexity**: Can become complex with many hook methods
- **Debugging**: Harder to trace execution flow across inheritance hierarchy
- **Limited Flexibility**: Cannot easily compose different algorithms

## Integration Tips

### 1. Strategy Pattern Integration
```go
// Combine Template Method with Strategy for flexible steps
type TemplateWithStrategy struct {
    validationStrategy ValidationStrategy
    processingStrategy ProcessingStrategy
}

func (tws *TemplateWithStrategy) processTemplate() error {
    // Template method steps
    if err := tws.preProcess(); err != nil {
        return err
    }
    
    // Use strategy for validation
    if err := tws.validationStrategy.Validate(); err != nil {
        return err
    }
    
    // Use strategy for processing
    return tws.processingStrategy.Process()
}
```

### 2. Observer Pattern Integration
```go
type ObservableTemplate struct {
    *BaseTemplate
    observers []TemplateObserver
}

type TemplateObserver interface {
    OnStepStarted(stepName string)
    OnStepCompleted(stepName string, result interface{})
    OnStepFailed(stepName string, err error)
}

func (ot *ObservableTemplate) executeStep(stepName string, step func() (interface{}, error)) (interface{}, error) {
    // Notify step started
    for _, observer := range ot.observers {
        observer.OnStepStarted(stepName)
    }
    
    result, err := step()
    
    if err != nil {
        for _, observer := range ot.observers {
            observer.OnStepFailed(stepName, err)
        }
        return nil, err
    }
    
    // Notify step completed
    for _, observer := range ot.observers {
        observer.OnStepCompleted(stepName, result)
    }
    
    return result, nil
}
```

### 3. Decorator Pattern Integration
```go
type TemplateDecorator interface {
    Decorate(base TemplateMethod) TemplateMethod
}

type LoggingDecorator struct {
    logger *zap.Logger
}

func (ld *LoggingDecorator) Decorate(base TemplateMethod) TemplateMethod {
    return &LoggingTemplate{
        base:   base,
        logger: ld.logger,
    }
}

type LoggingTemplate struct {
    base   TemplateMethod
    logger *zap.Logger
}

func (lt *LoggingTemplate) Execute() error {
    lt.logger.Info("Template execution started")
    
    err := lt.base.Execute()
    
    if err != nil {
        lt.logger.Error("Template execution failed", zap.Error(err))
    } else {
        lt.logger.Info("Template execution completed successfully")
    }
    
    return err
}
```

## Common Interview Questions

### 1. **How does Template Method differ from Strategy pattern?**

**Answer:**

| Aspect | Template Method | Strategy |
|--------|-----------------|----------|
| **Purpose** | Define algorithm skeleton | Define algorithm family |
| **Flexibility** | Fixed structure, variable steps | Interchangeable algorithms |
| **Runtime** | Structure fixed at compile-time | Algorithm changeable at runtime |
| **Inheritance** | Uses inheritance heavily | Uses composition |
| **Control** | Parent controls flow | Client controls selection |

**Template Method Example:**
```go
func (processor *PaymentProcessor) processPayment() error {
    // Fixed algorithm structure
    processor.validate()    // Step 1 - always happens
    processor.authenticate() // Step 2 - always happens  
    processor.execute()     // Step 3 - always happens
    processor.notify()      // Step 4 - always happens
}

// Subclasses implement specific steps differently
func (cc *CreditCardProcessor) authenticate() error {
    // Credit card specific authentication
}
```

**Strategy Example:**
```go
type PaymentStrategy interface {
    ProcessPayment() error
}

func (processor *PaymentProcessor) SetStrategy(strategy PaymentStrategy) {
    processor.strategy = strategy
}

func (processor *PaymentProcessor) ProcessPayment() error {
    // Delegate entire algorithm to strategy
    return processor.strategy.ProcessPayment()
}

// Can change strategy at runtime
processor.SetStrategy(CreditCardStrategy{})
processor.SetStrategy(BankTransferStrategy{})
```

### 2. **How do you handle error scenarios in Template Method?**

**Answer:**

**1. Early Return Strategy:**
```go
func (base *BaseProcessor) processTemplate() error {
    if err := base.step1(); err != nil {
        base.handleError("step1", err)
        return err // Stop processing
    }
    
    if err := base.step2(); err != nil {
        base.handleError("step2", err)
        return err // Stop processing
    }
    
    return base.step3()
}
```

**2. Error Accumulation:**
```go
func (base *BaseProcessor) processTemplate() error {
    var errors []error
    
    if err := base.step1(); err != nil {
        errors = append(errors, err)
    }
    
    if err := base.step2(); err != nil {
        errors = append(errors, err)
    }
    
    if err := base.step3(); err != nil {
        errors = append(errors, err)
    }
    
    if len(errors) > 0 {
        return &MultiError{Errors: errors}
    }
    
    return nil
}
```

**3. Compensation Pattern:**
```go
func (base *BaseProcessor) processTemplate() error {
    var completedSteps []string
    
    if err := base.step1(); err != nil {
        return err
    }
    completedSteps = append(completedSteps, "step1")
    
    if err := base.step2(); err != nil {
        base.compensate(completedSteps)
        return err
    }
    completedSteps = append(completedSteps, "step2")
    
    if err := base.step3(); err != nil {
        base.compensate(completedSteps)
        return err
    }
    
    return nil
}

func (base *BaseProcessor) compensate(steps []string) {
    // Undo completed steps in reverse order
    for i := len(steps) - 1; i >= 0; i-- {
        base.undoStep(steps[i])
    }
}
```

**4. Circuit Breaker Integration:**
```go
type CircuitBreakerTemplate struct {
    *BaseTemplate
    breakers map[string]*CircuitBreaker
}

func (cbt *CircuitBreakerTemplate) executeStep(stepName string, step func() error) error {
    breaker := cbt.breakers[stepName]
    if breaker == nil {
        breaker = NewCircuitBreaker(stepName)
        cbt.breakers[stepName] = breaker
    }
    
    return breaker.Execute(step)
}
```

### 3. **How do you make Template Method testable?**

**Answer:**

**1. Mock Hook Methods:**
```go
type TestableProcessor struct {
    *BaseProcessor
    mockValidation func() error
    mockExecution  func() error
}

func (tp *TestableProcessor) validate() error {
    if tp.mockValidation != nil {
        return tp.mockValidation()
    }
    return tp.BaseProcessor.validate()
}

func TestProcessorTemplate(t *testing.T) {
    processor := &TestableProcessor{}
    processor.mockValidation = func() error {
        return fmt.Errorf("validation failed")
    }
    
    err := processor.ProcessTemplate()
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "validation failed")
}
```

**2. Dependency Injection:**
```go
type PaymentProcessor struct {
    validator    Validator
    gateway      PaymentGateway
    notifier     Notifier
}

func NewPaymentProcessor(v Validator, g PaymentGateway, n Notifier) *PaymentProcessor {
    return &PaymentProcessor{
        validator: v,
        gateway:   g,
        notifier:  n,
    }
}

func TestPaymentProcessorWithMocks(t *testing.T) {
    mockValidator := &MockValidator{}
    mockGateway := &MockPaymentGateway{}
    mockNotifier := &MockNotifier{}
    
    processor := NewPaymentProcessor(mockValidator, mockGateway, mockNotifier)
    
    mockValidator.On("Validate", mock.Anything).Return(nil)
    mockGateway.On("Process", mock.Anything).Return(nil)
    mockNotifier.On("Notify", mock.Anything).Return(nil)
    
    err := processor.ProcessPayment(testRequest)
    assert.NoError(t, err)
    
    mockValidator.AssertExpectations(t)
    mockGateway.AssertExpectations(t)
    mockNotifier.AssertExpectations(t)
}
```

**3. Step Verification:**
```go
type VerifiableTemplate struct {
    *BaseTemplate
    executedSteps []string
}

func (vt *VerifiableTemplate) recordStep(stepName string) {
    vt.executedSteps = append(vt.executedSteps, stepName)
}

func (vt *VerifiableTemplate) step1() error {
    vt.recordStep("step1")
    return vt.BaseTemplate.step1()
}

func TestTemplateStepExecution(t *testing.T) {
    template := &VerifiableTemplate{}
    template.Execute()
    
    expectedSteps := []string{"step1", "step2", "step3"}
    assert.Equal(t, expectedSteps, template.executedSteps)
}
```

### 4. **When should you use Template Method vs Composition?**

**Answer:**

**Use Template Method when:**
- Algorithm structure is stable and well-defined
- Multiple implementations follow the same process
- You're building a framework for others to extend
- Steps have dependencies and must execute in order
- You want to enforce certain invariants

**Use Composition when:**
- Need runtime flexibility in algorithm structure
- Want to combine different behaviors dynamically
- Algorithm steps can be reused in different contexts
- Testing individual components is important
- Want to avoid inheritance coupling

**Template Method Example:**
```go
// Good for: Fixed payment processing flow
type PaymentProcessor struct {
    // Algorithm structure is fixed
}

func (pp *PaymentProcessor) ProcessPayment() error {
    pp.validate()     // Always step 1
    pp.authenticate() // Always step 2
    pp.execute()      // Always step 3
    pp.notify()       // Always step 4
    return nil
}
```

**Composition Example:**
```go
// Good for: Flexible pipeline construction
type PaymentPipeline struct {
    steps []PaymentStep
}

func (pp *PaymentPipeline) AddStep(step PaymentStep) {
    pp.steps = append(pp.steps, step)
}

func (pp *PaymentPipeline) Process() error {
    for _, step := range pp.steps {
        if err := step.Execute(); err != nil {
            return err
        }
    }
    return nil
}

// Can build different pipelines for different scenarios
domesticPipeline := NewPaymentPipeline()
domesticPipeline.AddStep(BasicValidation{})
domesticPipeline.AddStep(SimpleProcessing{})

internationalPipeline := NewPaymentPipeline()
internationalPipeline.AddStep(EnhancedValidation{})
internationalPipeline.AddStep(CurrencyConversion{})
internationalPipeline.AddStep(ComplianceCheck{})
internationalPipeline.AddStep(ComplexProcessing{})
```

### 5. **How do you implement Template Method without inheritance in Go?**

**Answer:**

Since Go doesn't have traditional inheritance, here are several approaches:

**1. Interface-based Approach:**
```go
type PaymentStep interface {
    Execute(ctx *PaymentContext) error
}

type PaymentTemplate struct {
    validator     PaymentStep
    authenticator PaymentStep
    processor     PaymentStep
    notifier      PaymentStep
}

func (pt *PaymentTemplate) ProcessPayment(ctx *PaymentContext) error {
    steps := []PaymentStep{
        pt.validator,
        pt.authenticator,
        pt.processor,
        pt.notifier,
    }
    
    for _, step := range steps {
        if err := step.Execute(ctx); err != nil {
            return err
        }
    }
    
    return nil
}

// Implementations
type CreditCardValidator struct{}
func (ccv CreditCardValidator) Execute(ctx *PaymentContext) error {
    // Credit card specific validation
    return nil
}

type BankTransferValidator struct{}
func (btv BankTransferValidator) Execute(ctx *PaymentContext) error {
    // Bank transfer specific validation
    return nil
}
```

**2. Function-based Approach:**
```go
type ProcessingStep func(*PaymentContext) error

type FunctionalTemplate struct {
    steps []ProcessingStep
}

func (ft *FunctionalTemplate) Process(ctx *PaymentContext) error {
    for _, step := range ft.steps {
        if err := step(ctx); err != nil {
            return err
        }
    }
    return nil
}

// Create different templates
func NewCreditCardTemplate() *FunctionalTemplate {
    return &FunctionalTemplate{
        steps: []ProcessingStep{
            validateCreditCard,
            authenticateCard,
            processCreditCardPayment,
            notifySuccess,
        },
    }
}

func NewBankTransferTemplate() *FunctionalTemplate {
    return &FunctionalTemplate{
        steps: []ProcessingStep{
            validateBankAccount,
            verifyFunds,
            processBankTransfer,
            notifySuccess,
        },
    }
}
```

**3. Embedded Struct Approach:**
```go
type BasePaymentProcessor struct {
    logger *zap.Logger
}

func (bpp *BasePaymentProcessor) ProcessPayment(processor PaymentProcessor) error {
    if err := processor.Validate(); err != nil {
        return err
    }
    
    if err := processor.Authenticate(); err != nil {
        return err
    }
    
    if err := processor.Execute(); err != nil {
        return err
    }
    
    return processor.Notify()
}

type PaymentProcessor interface {
    Validate() error
    Authenticate() error
    Execute() error
    Notify() error
}

type CreditCardProcessor struct {
    BasePaymentProcessor
}

func (ccp *CreditCardProcessor) Validate() error {
    // Credit card validation
    return nil
}

// Other methods...

// Usage
base := &BasePaymentProcessor{logger: logger}
processor := &CreditCardProcessor{BasePaymentProcessor: *base}
err := base.ProcessPayment(processor)
```
