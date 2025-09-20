# 💳 Advanced Fintech & Payment Systems

## Table of Contents
1. [Payment Processing Architecture](#payment-processing-architecture)
2. [Payment Gateways](#payment-gateways)
3. [Fraud Detection](#fraud-detection)
4. [Compliance & Regulations](#compliance--regulations)
5. [Settlement & Reconciliation](#settlement--reconciliation)
6. [Risk Management](#risk-management)
7. [Go Implementation Examples](#go-implementation-examples)
8. [Interview Questions](#interview-questions)

## Payment Processing Architecture

### Core Payment System

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type PaymentProcessor struct {
    gateways    map[string]PaymentGateway
    validators  []PaymentValidator
    mutex       sync.RWMutex
}

type PaymentRequest struct {
    ID          string
    Amount      int64
    Currency    string
    CardNumber  string
    CVV         string
    ExpiryDate  string
    MerchantID  string
    CustomerID  string
    Metadata    map[string]string
}

type PaymentResponse struct {
    ID            string
    Status        PaymentStatus
    TransactionID string
    GatewayID     string
    ErrorCode     string
    ErrorMessage  string
    ProcessedAt   time.Time
}

type PaymentStatus int

const (
    PENDING PaymentStatus = iota
    PROCESSING
    SUCCESS
    FAILED
    CANCELLED
    REFUNDED
)

type PaymentGateway interface {
    ProcessPayment(ctx context.Context, req *PaymentRequest) (*PaymentResponse, error)
    RefundPayment(ctx context.Context, transactionID string, amount int64) error
    GetPaymentStatus(ctx context.Context, transactionID string) (*PaymentResponse, error)
}

func NewPaymentProcessor() *PaymentProcessor {
    return &PaymentProcessor{
        gateways:   make(map[string]PaymentGateway),
        validators: make([]PaymentValidator, 0),
    }
}

func (pp *PaymentProcessor) AddGateway(name string, gateway PaymentGateway) {
    pp.mutex.Lock()
    defer pp.mutex.Unlock()
    pp.gateways[name] = gateway
}

func (pp *PaymentProcessor) AddValidator(validator PaymentValidator) {
    pp.mutex.Lock()
    defer pp.mutex.Unlock()
    pp.validators = append(pp.validators, validator)
}

func (pp *PaymentProcessor) ProcessPayment(ctx context.Context, req *PaymentRequest) (*PaymentResponse, error) {
    // Validate payment request
    for _, validator := range pp.validators {
        if err := validator.Validate(req); err != nil {
            return &PaymentResponse{
                ID:           req.ID,
                Status:       FAILED,
                ErrorCode:    "VALIDATION_ERROR",
                ErrorMessage: err.Error(),
                ProcessedAt:  time.Now(),
            }, err
        }
    }
    
    // Select appropriate gateway
    gateway := pp.selectGateway(req)
    if gateway == nil {
        return &PaymentResponse{
            ID:           req.ID,
            Status:       FAILED,
            ErrorCode:    "NO_GATEWAY",
            ErrorMessage: "No suitable payment gateway found",
            ProcessedAt:  time.Now(),
        }, fmt.Errorf("no suitable payment gateway found")
    }
    
    // Process payment
    resp, err := gateway.ProcessPayment(ctx, req)
    if err != nil {
        resp.Status = FAILED
        resp.ErrorMessage = err.Error()
    }
    
    resp.ProcessedAt = time.Now()
    return resp, err
}

func (pp *PaymentProcessor) selectGateway(req *PaymentRequest) PaymentGateway {
    pp.mutex.RLock()
    defer pp.mutex.RUnlock()
    
    // Simple gateway selection logic
    // In production, use more sophisticated routing
    for _, gateway := range pp.gateways {
        return gateway
    }
    
    return nil
}

type PaymentValidator interface {
    Validate(req *PaymentRequest) error
}

type CardValidator struct{}

func (cv *CardValidator) Validate(req *PaymentRequest) error {
    // Validate card number using Luhn algorithm
    if !cv.isValidCardNumber(req.CardNumber) {
        return fmt.Errorf("invalid card number")
    }
    
    // Validate CVV
    if len(req.CVV) < 3 || len(req.CVV) > 4 {
        return fmt.Errorf("invalid CVV")
    }
    
    // Validate expiry date
    if !cv.isValidExpiryDate(req.ExpiryDate) {
        return fmt.Errorf("invalid expiry date")
    }
    
    return nil
}

func (cv *CardValidator) isValidCardNumber(cardNumber string) bool {
    // Luhn algorithm implementation
    sum := 0
    alternate := false
    
    for i := len(cardNumber) - 1; i >= 0; i-- {
        digit := int(cardNumber[i] - '0')
        
        if alternate {
            digit *= 2
            if digit > 9 {
                digit = (digit % 10) + 1
            }
        }
        
        sum += digit
        alternate = !alternate
    }
    
    return sum%10 == 0
}

func (cv *CardValidator) isValidExpiryDate(expiryDate string) bool {
    // Parse MM/YY format
    if len(expiryDate) != 5 || expiryDate[2] != '/' {
        return false
    }
    
    month := expiryDate[:2]
    year := expiryDate[3:]
    
    if month < "01" || month > "12" {
        return false
    }
    
    currentYear := time.Now().Year() % 100
    if year < fmt.Sprintf("%02d", currentYear) {
        return false
    }
    
    return true
}
```

## Payment Gateways

### Gateway Implementation

```go
package main

import (
    "context"
    "crypto/hmac"
    "crypto/sha256"
    "encoding/hex"
    "fmt"
    "net/http"
    "time"
)

type RazorpayGateway struct {
    keyID     string
    keySecret string
    baseURL   string
    client    *http.Client
}

func NewRazorpayGateway(keyID, keySecret string) *RazorpayGateway {
    return &RazorpayGateway{
        keyID:     keyID,
        keySecret: keySecret,
        baseURL:   "https://api.razorpay.com/v1",
        client:    &http.Client{Timeout: 30 * time.Second},
    }
}

func (rg *RazorpayGateway) ProcessPayment(ctx context.Context, req *PaymentRequest) (*PaymentResponse, error) {
    // Create payment order
    order, err := rg.createOrder(ctx, req)
    if err != nil {
        return nil, err
    }
    
    // Process payment
    payment, err := rg.capturePayment(ctx, order.ID, req.Amount)
    if err != nil {
        return nil, err
    }
    
    return &PaymentResponse{
        ID:            req.ID,
        Status:        SUCCESS,
        TransactionID: payment.ID,
        GatewayID:     "razorpay",
        ProcessedAt:   time.Now(),
    }, nil
}

func (rg *RazorpayGateway) createOrder(ctx context.Context, req *PaymentRequest) (*Order, error) {
    orderReq := map[string]interface{}{
        "amount":   req.Amount,
        "currency": req.Currency,
        "receipt":  req.ID,
    }
    
    // Make API call to Razorpay
    // Implementation would make actual HTTP request
    return &Order{
        ID:     "order_" + req.ID,
        Amount: req.Amount,
        Status: "created",
    }, nil
}

func (rg *RazorpayGateway) capturePayment(ctx context.Context, orderID string, amount int64) (*Payment, error) {
    // Implementation would capture payment
    return &Payment{
        ID:     "pay_" + orderID,
        Amount: amount,
        Status: "captured",
    }, nil
}

func (rg *RazorpayGateway) RefundPayment(ctx context.Context, transactionID string, amount int64) error {
    // Implementation would process refund
    return nil
}

func (rg *RazorpayGateway) GetPaymentStatus(ctx context.Context, transactionID string) (*PaymentResponse, error) {
    // Implementation would fetch payment status
    return &PaymentResponse{
        ID:            transactionID,
        Status:        SUCCESS,
        TransactionID: transactionID,
        GatewayID:     "razorpay",
        ProcessedAt:   time.Now(),
    }, nil
}

type Order struct {
    ID     string
    Amount int64
    Status string
}

type Payment struct {
    ID     string
    Amount int64
    Status string
}

// Stripe Gateway
type StripeGateway struct {
    secretKey string
    baseURL   string
    client    *http.Client
}

func NewStripeGateway(secretKey string) *StripeGateway {
    return &StripeGateway{
        secretKey: secretKey,
        baseURL:   "https://api.stripe.com/v1",
        client:    &http.Client{Timeout: 30 * time.Second},
    }
}

func (sg *StripeGateway) ProcessPayment(ctx context.Context, req *PaymentRequest) (*PaymentResponse, error) {
    // Create payment intent
    intent, err := sg.createPaymentIntent(ctx, req)
    if err != nil {
        return nil, err
    }
    
    // Confirm payment
    confirmed, err := sg.confirmPayment(ctx, intent.ID)
    if err != nil {
        return nil, err
    }
    
    return &PaymentResponse{
        ID:            req.ID,
        Status:        SUCCESS,
        TransactionID: confirmed.ID,
        GatewayID:     "stripe",
        ProcessedAt:   time.Now(),
    }, nil
}

func (sg *StripeGateway) createPaymentIntent(ctx context.Context, req *PaymentRequest) (*PaymentIntent, error) {
    // Implementation would create payment intent
    return &PaymentIntent{
        ID:     "pi_" + req.ID,
        Amount: req.Amount,
        Status: "requires_confirmation",
    }, nil
}

func (sg *StripeGateway) confirmPayment(ctx context.Context, intentID string) (*PaymentIntent, error) {
    // Implementation would confirm payment
    return &PaymentIntent{
        ID:     intentID,
        Status: "succeeded",
    }, nil
}

func (sg *StripeGateway) RefundPayment(ctx context.Context, transactionID string, amount int64) error {
    // Implementation would process refund
    return nil
}

func (sg *StripeGateway) GetPaymentStatus(ctx context.Context, transactionID string) (*PaymentResponse, error) {
    // Implementation would fetch payment status
    return &PaymentResponse{
        ID:            transactionID,
        Status:        SUCCESS,
        TransactionID: transactionID,
        GatewayID:     "stripe",
        ProcessedAt:   time.Now(),
    }, nil
}

type PaymentIntent struct {
    ID     string
    Amount int64
    Status string
}
```

## Fraud Detection

### ML-based Fraud Detection

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type FraudDetector struct {
    rules        []FraudRule
    mlModel      *FraudMLModel
    riskScores   map[string]float64
    mutex        sync.RWMutex
}

type FraudRule struct {
    Name        string
    Description string
    RiskScore   float64
    Evaluate    func(req *PaymentRequest) bool
}

type FraudMLModel struct {
    features []string
    weights  map[string]float64
    threshold float64
}

func NewFraudDetector() *FraudDetector {
    return &FraudDetector{
        rules:      make([]FraudRule, 0),
        mlModel:    &FraudMLModel{},
        riskScores: make(map[string]float64),
    }
}

func (fd *FraudDetector) AddRule(rule FraudRule) {
    fd.mutex.Lock()
    defer fd.mutex.Unlock()
    fd.rules = append(fd.rules, rule)
}

func (fd *FraudDetector) DetectFraud(ctx context.Context, req *PaymentRequest) (*FraudResult, error) {
    fd.mutex.Lock()
    defer fd.mutex.Unlock()
    
    totalRiskScore := 0.0
    triggeredRules := make([]string, 0)
    
    // Evaluate rules
    for _, rule := range fd.rules {
        if rule.Evaluate(req) {
            totalRiskScore += rule.RiskScore
            triggeredRules = append(triggeredRules, rule.Name)
        }
    }
    
    // ML model evaluation
    mlScore := fd.mlModel.Evaluate(req)
    totalRiskScore += mlScore
    
    // Determine fraud status
    isFraud := totalRiskScore > 0.7
    
    result := &FraudResult{
        IsFraud:        isFraud,
        RiskScore:      totalRiskScore,
        TriggeredRules: triggeredRules,
        MLScore:        mlScore,
        Timestamp:      time.Now(),
    }
    
    // Store risk score
    fd.riskScores[req.ID] = totalRiskScore
    
    return result, nil
}

type FraudResult struct {
    IsFraud        bool
    RiskScore      float64
    TriggeredRules []string
    MLScore        float64
    Timestamp      time.Time
}

func (fml *FraudMLModel) Evaluate(req *PaymentRequest) float64 {
    // Simplified ML model evaluation
    score := 0.0
    
    // Amount-based scoring
    if req.Amount > 100000 { // > $1000
        score += 0.2
    }
    
    // Time-based scoring
    hour := time.Now().Hour()
    if hour < 6 || hour > 22 {
        score += 0.1
    }
    
    // Card number patterns
    if len(req.CardNumber) != 16 {
        score += 0.3
    }
    
    return score
}

// Common fraud rules
func CreateCommonFraudRules() []FraudRule {
    return []FraudRule{
        {
            Name:        "High Amount",
            Description: "Transaction amount exceeds threshold",
            RiskScore:   0.3,
            Evaluate: func(req *PaymentRequest) bool {
                return req.Amount > 50000 // > $500
            },
        },
        {
            Name:        "Suspicious Time",
            Description: "Transaction during suspicious hours",
            RiskScore:   0.2,
            Evaluate: func(req *PaymentRequest) bool {
                hour := time.Now().Hour()
                return hour < 2 || hour > 23
            },
        },
        {
            Name:        "Rapid Transactions",
            Description: "Multiple transactions in short time",
            RiskScore:   0.4,
            Evaluate: func(req *PaymentRequest) bool {
                // Implementation would check transaction history
                return false
            },
        },
    }
}
```

## Compliance & Regulations

### PCI DSS Compliance

```go
package main

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "fmt"
    "io"
)

type PCICompliance struct {
    encryptionKey []byte
}

func NewPCICompliance(encryptionKey []byte) *PCICompliance {
    return &PCICompliance{
        encryptionKey: encryptionKey,
    }
}

func (pci *PCICompliance) EncryptCardData(cardNumber, cvv string) (string, string, error) {
    encryptedCard, err := pci.encrypt(cardNumber)
    if err != nil {
        return "", "", err
    }
    
    encryptedCVV, err := pci.encrypt(cvv)
    if err != nil {
        return "", "", err
    }
    
    return encryptedCard, encryptedCVV, nil
}

func (pci *PCICompliance) DecryptCardData(encryptedCard, encryptedCVV string) (string, string, error) {
    cardNumber, err := pci.decrypt(encryptedCard)
    if err != nil {
        return "", "", err
    }
    
    cvv, err := pci.decrypt(encryptedCVV)
    if err != nil {
        return "", "", err
    }
    
    return cardNumber, cvv, nil
}

func (pci *PCICompliance) encrypt(plaintext string) (string, error) {
    block, err := aes.NewCipher(pci.encryptionKey)
    if err != nil {
        return "", err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return "", err
    }
    
    nonce := make([]byte, gcm.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return "", err
    }
    
    ciphertext := gcm.Seal(nonce, nonce, []byte(plaintext), nil)
    return fmt.Sprintf("%x", ciphertext), nil
}

func (pci *PCICompliance) decrypt(ciphertext string) (string, error) {
    data, err := hex.DecodeString(ciphertext)
    if err != nil {
        return "", err
    }
    
    block, err := aes.NewCipher(pci.encryptionKey)
    if err != nil {
        return "", err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return "", err
    }
    
    nonceSize := gcm.NonceSize()
    if len(data) < nonceSize {
        return "", fmt.Errorf("ciphertext too short")
    }
    
    nonce, ciphertext := data[:nonceSize], data[nonceSize:]
    plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
    if err != nil {
        return "", err
    }
    
    return string(plaintext), nil
}

// KYC (Know Your Customer) Compliance
type KYCManager struct {
    validators []KYCValidator
}

type KYCValidator interface {
    Validate(customer *Customer) error
}

type Customer struct {
    ID        string
    Name      string
    Email     string
    Phone     string
    Address   string
    Documents []Document
    Status    KYCStatus
}

type Document struct {
    Type     string
    Number   string
    IssuedBy string
    Expiry   time.Time
}

type KYCStatus int

const (
    PENDING KYCStatus = iota
    VERIFIED
    REJECTED
    EXPIRED
)

func NewKYCManager() *KYCManager {
    return &KYCManager{
        validators: make([]KYCValidator, 0),
    }
}

func (kyc *KYCManager) AddValidator(validator KYCValidator) {
    kyc.validators = append(kyc.validators, validator)
}

func (kyc *KYCManager) VerifyCustomer(customer *Customer) error {
    for _, validator := range kyc.validators {
        if err := validator.Validate(customer); err != nil {
            customer.Status = REJECTED
            return err
        }
    }
    
    customer.Status = VERIFIED
    return nil
}

type EmailValidator struct{}

func (ev *EmailValidator) Validate(customer *Customer) error {
    // Basic email validation
    if customer.Email == "" {
        return fmt.Errorf("email is required")
    }
    
    // Check email format
    if !strings.Contains(customer.Email, "@") {
        return fmt.Errorf("invalid email format")
    }
    
    return nil
}

type PhoneValidator struct{}

func (pv *PhoneValidator) Validate(customer *Customer) error {
    // Basic phone validation
    if customer.Phone == "" {
        return fmt.Errorf("phone is required")
    }
    
    // Check phone format
    if len(customer.Phone) < 10 {
        return fmt.Errorf("invalid phone number")
    }
    
    return nil
}
```

## Settlement & Reconciliation

### Settlement System

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type SettlementManager struct {
    settlements map[string]*Settlement
    mutex       sync.RWMutex
}

type Settlement struct {
    ID            string
    MerchantID    string
    Amount        int64
    Currency      string
    Status        SettlementStatus
    CreatedAt     time.Time
    ProcessedAt   time.Time
    Transactions  []string
}

type SettlementStatus int

const (
    PENDING SettlementStatus = iota
    PROCESSING
    COMPLETED
    FAILED
)

func NewSettlementManager() *SettlementManager {
    return &SettlementManager{
        settlements: make(map[string]*Settlement),
    }
}

func (sm *SettlementManager) CreateSettlement(merchantID string, transactions []string) (*Settlement, error) {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()
    
    settlementID := fmt.Sprintf("settlement_%d", time.Now().Unix())
    
    settlement := &Settlement{
        ID:           settlementID,
        MerchantID:   merchantID,
        Status:       PENDING,
        CreatedAt:    time.Now(),
        Transactions: transactions,
    }
    
    sm.settlements[settlementID] = settlement
    return settlement, nil
}

func (sm *SettlementManager) ProcessSettlement(settlementID string) error {
    sm.mutex.Lock()
    defer sm.mutex.Unlock()
    
    settlement, exists := sm.settlements[settlementID]
    if !exists {
        return fmt.Errorf("settlement not found")
    }
    
    settlement.Status = PROCESSING
    
    // Calculate total amount
    totalAmount := int64(0)
    for _, transactionID := range settlement.Transactions {
        // In production, fetch transaction details
        totalAmount += 1000 // Simplified
    }
    
    settlement.Amount = totalAmount
    settlement.Status = COMPLETED
    settlement.ProcessedAt = time.Now()
    
    return nil
}

// Reconciliation System
type ReconciliationManager struct {
    transactions map[string]*Transaction
    mutex        sync.RWMutex
}

type Transaction struct {
    ID          string
    Amount      int64
    Currency    string
    Status      string
    GatewayID   string
    ProcessedAt time.Time
}

func NewReconciliationManager() *ReconciliationManager {
    return &ReconciliationManager{
        transactions: make(map[string]*Transaction),
    }
}

func (rm *ReconciliationManager) AddTransaction(transaction *Transaction) {
    rm.mutex.Lock()
    defer rm.mutex.Unlock()
    rm.transactions[transaction.ID] = transaction
}

func (rm *ReconciliationManager) Reconcile(gatewayTransactions []*Transaction) (*ReconciliationResult, error) {
    rm.mutex.RLock()
    defer rm.mutex.RUnlock()
    
    result := &ReconciliationResult{
        Matched:     make([]string, 0),
        Unmatched:   make([]string, 0),
        Discrepancies: make([]Discrepancy, 0),
    }
    
    // Find matches
    for _, gatewayTx := range gatewayTransactions {
        if internalTx, exists := rm.transactions[gatewayTx.ID]; exists {
            if rm.compareTransactions(internalTx, gatewayTx) {
                result.Matched = append(result.Matched, gatewayTx.ID)
            } else {
                result.Discrepancies = append(result.Discrepancies, Discrepancy{
                    TransactionID: gatewayTx.ID,
                    Field:         "amount",
                    InternalValue: internalTx.Amount,
                    GatewayValue:  gatewayTx.Amount,
                })
            }
        } else {
            result.Unmatched = append(result.Unmatched, gatewayTx.ID)
        }
    }
    
    return result, nil
}

func (rm *ReconciliationManager) compareTransactions(internal, gateway *Transaction) bool {
    return internal.Amount == gateway.Amount &&
           internal.Currency == gateway.Currency &&
           internal.Status == gateway.Status
}

type ReconciliationResult struct {
    Matched       []string
    Unmatched     []string
    Discrepancies []Discrepancy
}

type Discrepancy struct {
    TransactionID string
    Field         string
    InternalValue interface{}
    GatewayValue  interface{}
}
```

## Interview Questions

### Basic Concepts
1. **How do payment gateways work?**
2. **What is PCI DSS compliance?**
3. **How do you implement fraud detection?**
4. **What are the challenges in payment processing?**
5. **How do you handle payment failures?**

### Advanced Topics
1. **How would you design a payment system for high volume?**
2. **How do you implement real-time fraud detection?**
3. **What are the security considerations in payment systems?**
4. **How do you handle payment disputes and chargebacks?**
5. **How do you ensure compliance with financial regulations?**

### System Design
1. **Design a payment processing system.**
2. **How would you implement fraud detection at scale?**
3. **Design a settlement and reconciliation system.**
4. **How would you handle payment routing?**
5. **Design a compliance monitoring system.**

## Conclusion

Advanced fintech and payment systems require deep understanding of financial regulations, security, and scalability. Key areas to master:

- **Payment Processing**: Gateways, routing, processing
- **Fraud Detection**: ML models, rule engines, real-time detection
- **Compliance**: PCI DSS, KYC, AML, regulatory requirements
- **Security**: Encryption, tokenization, secure communication
- **Settlement**: Reconciliation, dispute handling, reporting
- **Risk Management**: Fraud prevention, chargeback handling

Understanding these concepts helps in:
- Building secure payment systems
- Implementing fraud detection
- Ensuring regulatory compliance
- Handling high-volume transactions
- Preparing for fintech interviews

This guide provides a comprehensive foundation for advanced fintech concepts and their practical implementation in Go.


## Risk Management

<!-- AUTO-GENERATED ANCHOR: originally referenced as #risk-management -->

Placeholder content. Please replace with proper section.


## Go Implementation Examples

<!-- AUTO-GENERATED ANCHOR: originally referenced as #go-implementation-examples -->

Placeholder content. Please replace with proper section.
