---
# Auto-generated front matter
Title: Code Quality Guide
LastUpdated: 2025-11-06T20:45:58.275968
Tags: []
Status: draft
---

# 📋 Code Quality & Review Guide

> **Comprehensive guide to code quality, review practices, and engineering best practices**

## 🎯 **Overview**

Code quality and review practices are crucial for senior engineers and technical leaders. This guide covers modern code quality practices, review techniques, and engineering standards expected in technical interviews and professional development.

## 📚 **Table of Contents**

1. [Code Quality Fundamentals](#code-quality-fundamentals)
2. [Code Review Process](#code-review-process)
3. [Static Analysis & Linting](#static-analysis--linting)
4. [Code Metrics & Standards](#code-metrics--standards)
5. [Refactoring Strategies](#refactoring-strategies)
6. [Technical Debt Management](#technical-debt-management)
7. [Security Best Practices](#security-best-practices)
8. [Performance Optimization](#performance-optimization)
9. [Documentation Standards](#documentation-standards)
10. [Team Practices & Culture](#team-practices--culture)
11. [Interview Questions](#interview-questions)

---

## 🏗️ **Code Quality Fundamentals**

### **SOLID Principles Implementation**

```go
// Single Responsibility Principle (SRP)
// ❌ Bad: Class with multiple responsibilities
type PaymentProcessor struct {
    db     *sql.DB
    logger *log.Logger
    config *Config
}

func (p *PaymentProcessor) ProcessPayment(payment *Payment) error {
    // Validation logic
    if payment.Amount <= 0 {
        return errors.New("invalid amount")
    }
    
    // Database operations
    _, err := p.db.Exec("INSERT INTO payments...", payment.ID, payment.Amount)
    if err != nil {
        return err
    }
    
    // Logging
    p.logger.Printf("Payment processed: %s", payment.ID)
    
    // Email notification
    emailService := NewEmailService(p.config.SMTPHost)
    return emailService.SendConfirmation(payment.UserEmail)
}

// ✅ Good: Separated responsibilities
type PaymentValidator interface {
    Validate(payment *Payment) error
}

type PaymentRepository interface {
    Save(payment *Payment) error
}

type NotificationService interface {
    SendConfirmation(email string, payment *Payment) error
}

type PaymentProcessor struct {
    validator   PaymentValidator
    repository  PaymentRepository
    notifier    NotificationService
    logger      *log.Logger
}

func (p *PaymentProcessor) ProcessPayment(payment *Payment) error {
    if err := p.validator.Validate(payment); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }
    
    if err := p.repository.Save(payment); err != nil {
        return fmt.Errorf("save failed: %w", err)
    }
    
    p.logger.Printf("Payment processed: %s", payment.ID)
    
    // Async notification
    go func() {
        if err := p.notifier.SendConfirmation(payment.UserEmail, payment); err != nil {
            p.logger.Printf("Notification failed: %v", err)
        }
    }()
    
    return nil
}

// Open/Closed Principle (OCP)
// ✅ Good: Open for extension, closed for modification
type PaymentMethod interface {
    Process(amount float64) (*PaymentResult, error)
    GetFee(amount float64) float64
}

type CreditCardPayment struct {
    cardNumber string
    cvv        string
}

func (c *CreditCardPayment) Process(amount float64) (*PaymentResult, error) {
    // Credit card processing logic
    return &PaymentResult{Success: true, TransactionID: "cc_123"}, nil
}

func (c *CreditCardPayment) GetFee(amount float64) float64 {
    return amount * 0.029 // 2.9% fee
}

type BankTransfer struct {
    accountNumber string
    routingNumber string
}

func (b *BankTransfer) Process(amount float64) (*PaymentResult, error) {
    // Bank transfer processing logic
    return &PaymentResult{Success: true, TransactionID: "bt_123"}, nil
}

func (b *BankTransfer) GetFee(amount float64) float64 {
    return 0.50 // Flat fee
}

// Payment processor can handle any payment method without modification
type UniversalPaymentProcessor struct {
    methods map[string]PaymentMethod
}

func (p *UniversalPaymentProcessor) ProcessPayment(methodType string, amount float64) (*PaymentResult, error) {
    method, exists := p.methods[methodType]
    if !exists {
        return nil, fmt.Errorf("unsupported payment method: %s", methodType)
    }
    
    return method.Process(amount)
}

// Liskov Substitution Principle (LSP)
// ✅ Good: Subtypes are substitutable for their base types
type Shape interface {
    Area() float64
    Perimeter() float64
}

type Rectangle struct {
    width, height float64
}

func (r Rectangle) Area() float64 {
    return r.width * r.height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.width + r.height)
}

type Square struct {
    side float64
}

func (s Square) Area() float64 {
    return s.side * s.side
}

func (s Square) Perimeter() float64 {
    return 4 * s.side
}

// Both Rectangle and Square can be used interchangeably
func CalculateShapeInfo(shape Shape) (float64, float64) {
    return shape.Area(), shape.Perimeter()
}

// Interface Segregation Principle (ISP)
// ✅ Good: Clients depend only on interfaces they use
type Reader interface {
    Read([]byte) (int, error)
}

type Writer interface {
    Write([]byte) (int, error)
}

type Closer interface {
    Close() error
}

// Specific interfaces for specific needs
type FileReader struct {
    file *os.File
}

func (f *FileReader) Read(data []byte) (int, error) {
    return f.file.Read(data)
}

// Dependency Inversion Principle (DIP)
// ✅ Good: Depend on abstractions, not concretions
type Database interface {
    Save(data interface{}) error
    Find(id string) (interface{}, error)
}

type UserService struct {
    db Database // Depends on abstraction
}

func (s *UserService) CreateUser(user *User) error {
    return s.db.Save(user)
}

// Concrete implementations
type PostgreSQLDatabase struct {
    conn *sql.DB
}

func (p *PostgreSQLDatabase) Save(data interface{}) error {
    // PostgreSQL specific implementation
    return nil
}

func (p *PostgreSQLDatabase) Find(id string) (interface{}, error) {
    // PostgreSQL specific implementation
    return nil, nil
}
```

### **Clean Code Principles**

```go
// ❌ Bad: Unclear naming and complex function
func p(u string, a float64, c string) (string, error) {
    if a <= 0 {
        return "", errors.New("invalid")
    }
    
    if c != "USD" && c != "EUR" && c != "GBP" {
        return "", errors.New("invalid currency")
    }
    
    id := fmt.Sprintf("pay_%d", time.Now().UnixNano())
    
    // Complex business logic mixed together
    if a > 10000 {
        // High value transaction
        if err := performKYCCheck(u); err != nil {
            return "", err
        }
        
        if err := checkFraudRules(u, a); err != nil {
            return "", err
        }
    }
    
    return id, nil
}

// ✅ Good: Clear naming and separated concerns
type PaymentRequest struct {
    UserID   string
    Amount   float64
    Currency string
}

type PaymentValidator struct {
    supportedCurrencies map[string]bool
    kycService         KYCService
    fraudService       FraudService
}

func (v *PaymentValidator) ValidatePaymentRequest(req PaymentRequest) error {
    if err := v.validateAmount(req.Amount); err != nil {
        return fmt.Errorf("amount validation failed: %w", err)
    }
    
    if err := v.validateCurrency(req.Currency); err != nil {
        return fmt.Errorf("currency validation failed: %w", err)
    }
    
    if v.isHighValueTransaction(req.Amount) {
        if err := v.performComplianceChecks(req); err != nil {
            return fmt.Errorf("compliance check failed: %w", err)
        }
    }
    
    return nil
}

func (v *PaymentValidator) validateAmount(amount float64) error {
    if amount <= 0 {
        return errors.New("amount must be positive")
    }
    
    if amount > MaxTransactionAmount {
        return errors.New("amount exceeds maximum limit")
    }
    
    return nil
}

func (v *PaymentValidator) validateCurrency(currency string) error {
    if !v.supportedCurrencies[currency] {
        return fmt.Errorf("unsupported currency: %s", currency)
    }
    
    return nil
}

func (v *PaymentValidator) isHighValueTransaction(amount float64) bool {
    return amount > HighValueThreshold
}

func (v *PaymentValidator) performComplianceChecks(req PaymentRequest) error {
    if err := v.kycService.VerifyUser(req.UserID); err != nil {
        return fmt.Errorf("KYC verification failed: %w", err)
    }
    
    if err := v.fraudService.CheckTransaction(req.UserID, req.Amount); err != nil {
        return fmt.Errorf("fraud check failed: %w", err)
    }
    
    return nil
}

func GeneratePaymentID() string {
    return fmt.Sprintf("pay_%d", time.Now().UnixNano())
}
```

---

## 🔍 **Code Review Process**

### **Code Review Checklist**

```go
// Code Review Checklist Template
type CodeReviewChecklist struct {
    // Functionality
    DoesCodeWork                bool
    MeetsRequirements          bool
    HandlesEdgeCases           bool
    HandlesErrors              bool
    
    // Code Quality
    IsReadable                 bool
    IsWellStructured           bool
    FollowsNamingConventions   bool
    HasAppropriateComments     bool
    
    // Performance
    IsPerformant               bool
    NoMemoryLeaks              bool
    EfficientAlgorithms        bool
    
    // Security
    NoSecurityVulnerabilities  bool
    InputValidation            bool
    OutputSanitization         bool
    AuthenticationChecks       bool
    
    // Testing
    HasUnitTests               bool
    HasIntegrationTests        bool
    TestCoverageAdequate       bool
    TestsAreReliable           bool
    
    // Documentation
    HasDocumentation           bool
    APIDocumentationUpdated    bool
    ReadmeUpdated              bool
}
```

### **Review Comments Examples**

```go
// ❌ Bad Review Comments
// "This is wrong"
// "Fix this"
// "Bad code"

// ✅ Good Review Comments
// "Consider using a context.Context here to handle timeouts properly:
//  func ProcessPayment(ctx context.Context, payment *Payment) error"
//
// "This could lead to a race condition. Consider using sync.Mutex:
//  type PaymentCache struct {
//      mu    sync.RWMutex
//      cache map[string]*Payment
//  }"
//
// "For better error handling, wrap the error with context:
//  return fmt.Errorf('failed to save payment %s: %w', payment.ID, err)"

// Example of constructive review
func ProcessPayment(payment *Payment) error {  // 🔍 Review: Missing context parameter
    if payment.Amount <= 0 {  // 🔍 Review: Consider extracting validation
        return errors.New("invalid amount")  // 🔍 Review: Use domain-specific errors
    }
    
    // 🔍 Review: No input sanitization
    query := fmt.Sprintf("INSERT INTO payments VALUES ('%s', %f)", 
        payment.ID, payment.Amount)  // 🔍 Review: SQL injection vulnerability
    
    _, err := db.Exec(query)  // 🔍 Review: No timeout handling
    return err  // 🔍 Review: Error lacks context
}

// ✅ Improved version after review
func ProcessPayment(ctx context.Context, payment *Payment) error {
    if err := validatePayment(payment); err != nil {
        return fmt.Errorf("payment validation failed: %w", err)
    }
    
    // Use parameterized query to prevent SQL injection
    query := "INSERT INTO payments (id, amount, currency, user_id) VALUES ($1, $2, $3, $4)"
    
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()
    
    _, err := db.ExecContext(ctx, query, payment.ID, payment.Amount, payment.Currency, payment.UserID)
    if err != nil {
        return fmt.Errorf("failed to save payment %s: %w", payment.ID, err)
    }
    
    return nil
}

func validatePayment(payment *Payment) error {
    if payment == nil {
        return ErrPaymentNil
    }
    
    if payment.Amount <= 0 {
        return ErrInvalidAmount
    }
    
    if payment.ID == "" {
        return ErrPaymentIDRequired
    }
    
    return nil
}

// Domain-specific errors
var (
    ErrPaymentNil        = errors.New("payment cannot be nil")
    ErrInvalidAmount     = errors.New("payment amount must be positive")
    ErrPaymentIDRequired = errors.New("payment ID is required")
)
```

### **Pull Request Template**

```markdown
## Pull Request Template

### Description
Brief description of changes and motivation.

### Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [ ] Documentation update

### Changes Made
- List of specific changes
- Include any architectural decisions
- Mention any new dependencies

### Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] Performance testing (if applicable)

### Security
- [ ] No sensitive data exposed
- [ ] Input validation implemented
- [ ] Authentication/authorization checked
- [ ] SQL injection prevention verified

### Documentation
- [ ] Code comments added/updated
- [ ] API documentation updated
- [ ] README updated (if applicable)
- [ ] Migration guide created (if breaking change)

### Checklist
- [ ] Code follows the project's style guidelines
- [ ] Self-review completed
- [ ] Code is well-documented
- [ ] Tests pass locally
- [ ] No merge conflicts
- [ ] Backward compatibility maintained (unless breaking change)

### Screenshots/GIFs
(If applicable, add screenshots or GIFs to help explain your changes)

### Related Issues
Closes #issue_number
```

---

## 🔧 **Static Analysis & Linting**

### **Go Static Analysis Setup**

```go
// .golangci.yml configuration
package main

import (
    "context"
    "fmt"
    "time"
)

// Example of code that passes static analysis
type PaymentService struct {
    repository PaymentRepository
    validator  PaymentValidator
    logger     Logger
}

// NewPaymentService creates a new payment service instance
func NewPaymentService(repo PaymentRepository, validator PaymentValidator, logger Logger) *PaymentService {
    if repo == nil {
        panic("repository cannot be nil")
    }
    if validator == nil {
        panic("validator cannot be nil")
    }
    if logger == nil {
        panic("logger cannot be nil")
    }
    
    return &PaymentService{
        repository: repo,
        validator:  validator,
        logger:     logger,
    }
}

// ProcessPayment processes a payment with proper error handling
func (s *PaymentService) ProcessPayment(ctx context.Context, req *PaymentRequest) (*Payment, error) {
    if req == nil {
        return nil, fmt.Errorf("payment request cannot be nil")
    }
    
    // Validate request
    if err := s.validator.Validate(req); err != nil {
        s.logger.Error("validation failed", "error", err, "request", req)
        return nil, fmt.Errorf("validation failed: %w", err)
    }
    
    // Create payment
    payment := &Payment{
        ID:        generatePaymentID(),
        UserID:    req.UserID,
        Amount:    req.Amount,
        Currency:  req.Currency,
        Status:    StatusPending,
        CreatedAt: time.Now().UTC(),
    }
    
    // Save to repository
    if err := s.repository.Save(ctx, payment); err != nil {
        s.logger.Error("failed to save payment", "error", err, "payment", payment)
        return nil, fmt.Errorf("failed to save payment: %w", err)
    }
    
    s.logger.Info("payment processed successfully", "paymentID", payment.ID)
    return payment, nil
}

func generatePaymentID() string {
    return fmt.Sprintf("pay_%d", time.Now().UnixNano())
}

// Static analysis rules configuration
/*
.golangci.yml:

run:
  timeout: 5m
  tests: true

linters-settings:
  govet:
    check-shadowing: true
  golint:
    min-confidence: 0
  gocyclo:
    min-complexity: 10
  maligned:
    suggest-new: true
  dupl:
    threshold: 100
  goconst:
    min-len: 2
    min-occurrences: 2
  misspell:
    locale: US
  lll:
    line-length: 140
  goimports:
    local-prefixes: github.com/company/project
  gocritic:
    enabled-tags:
      - diagnostic
      - experimental
      - opinionated
      - performance
      - style

linters:
  enable:
    - megacheck
    - govet
    - gocyclo
    - golint
    - misspell
    - goconst
    - goimports
    - gosec
    - ineffassign
    - interfacer
    - unconvert
    - gocritic
    - gofmt
    - goimports
  disable:
    - typecheck

issues:
  exclude-rules:
    - path: _test\.go
      linters:
        - gocyclo
        - errcheck
        - dupl
        - gosec
*/
```

### **Pre-commit Hooks**

```bash
#!/bin/bash
# .git/hooks/pre-commit

set -e

echo "Running pre-commit hooks..."

# Format code
echo "Formatting Go code..."
gofmt -w .

# Run imports
echo "Organizing imports..."
goimports -w .

# Run linters
echo "Running golangci-lint..."
golangci-lint run

# Run tests
echo "Running tests..."
go test -short ./...

# Security scan
echo "Running security scan..."
gosec ./...

# Check for direct dependencies vulnerabilities
echo "Checking for vulnerabilities..."
govulncheck ./...

echo "All pre-commit checks passed!"
```

---

## 📊 **Code Metrics & Standards**

### **Complexity Metrics**

```go
// Cyclomatic Complexity Example
// ❌ Bad: High complexity function (CC = 8)
func ProcessTransaction(transaction *Transaction) error {
    if transaction == nil {
        return errors.New("transaction is nil")
    }
    
    if transaction.Amount <= 0 {
        return errors.New("invalid amount")
    }
    
    if transaction.Currency == "" {
        return errors.New("currency required")
    }
    
    if transaction.Type == "credit_card" {
        if transaction.CardNumber == "" {
            return errors.New("card number required")
        }
        if len(transaction.CardNumber) < 13 {
            return errors.New("invalid card number")
        }
        if transaction.CVV == "" {
            return errors.New("CVV required")
        }
    } else if transaction.Type == "bank_transfer" {
        if transaction.AccountNumber == "" {
            return errors.New("account number required")
        }
        if transaction.RoutingNumber == "" {
            return errors.New("routing number required")
        }
    } else if transaction.Type == "digital_wallet" {
        if transaction.WalletID == "" {
            return errors.New("wallet ID required")
        }
    } else {
        return errors.New("unsupported transaction type")
    }
    
    return nil
}

// ✅ Good: Reduced complexity using strategy pattern (CC = 2)
type TransactionValidator interface {
    Validate(transaction *Transaction) error
}

type CreditCardValidator struct{}

func (v *CreditCardValidator) Validate(transaction *Transaction) error {
    if transaction.CardNumber == "" {
        return errors.New("card number required")
    }
    if len(transaction.CardNumber) < 13 {
        return errors.New("invalid card number")
    }
    if transaction.CVV == "" {
        return errors.New("CVV required")
    }
    return nil
}

type BankTransferValidator struct{}

func (v *BankTransferValidator) Validate(transaction *Transaction) error {
    if transaction.AccountNumber == "" {
        return errors.New("account number required")
    }
    if transaction.RoutingNumber == "" {
        return errors.New("routing number required")
    }
    return nil
}

type DigitalWalletValidator struct{}

func (v *DigitalWalletValidator) Validate(transaction *Transaction) error {
    if transaction.WalletID == "" {
        return errors.New("wallet ID required")
    }
    return nil
}

type TransactionProcessor struct {
    validators map[string]TransactionValidator
}

func NewTransactionProcessor() *TransactionProcessor {
    return &TransactionProcessor{
        validators: map[string]TransactionValidator{
            "credit_card":    &CreditCardValidator{},
            "bank_transfer":  &BankTransferValidator{},
            "digital_wallet": &DigitalWalletValidator{},
        },
    }
}

func (p *TransactionProcessor) ProcessTransaction(transaction *Transaction) error {
    if err := p.validateBasicFields(transaction); err != nil {
        return err
    }
    
    validator, exists := p.validators[transaction.Type]
    if !exists {
        return fmt.Errorf("unsupported transaction type: %s", transaction.Type)
    }
    
    return validator.Validate(transaction)
}

func (p *TransactionProcessor) validateBasicFields(transaction *Transaction) error {
    if transaction == nil {
        return errors.New("transaction is nil")
    }
    if transaction.Amount <= 0 {
        return errors.New("invalid amount")
    }
    if transaction.Currency == "" {
        return errors.New("currency required")
    }
    return nil
}
```

### **Code Coverage Standards**

```go
// Test coverage example
package payment

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestTransactionProcessor_ProcessTransaction(t *testing.T) {
    processor := NewTransactionProcessor()
    
    testCases := []struct {
        name        string
        transaction *Transaction
        expectError bool
        errorMsg    string
    }{
        {
            name:        "nil transaction",
            transaction: nil,
            expectError: true,
            errorMsg:    "transaction is nil",
        },
        {
            name: "invalid amount",
            transaction: &Transaction{
                Amount:   -100,
                Currency: "USD",
                Type:     "credit_card",
            },
            expectError: true,
            errorMsg:    "invalid amount",
        },
        {
            name: "missing currency",
            transaction: &Transaction{
                Amount: 100,
                Type:   "credit_card",
            },
            expectError: true,
            errorMsg:    "currency required",
        },
        {
            name: "valid credit card transaction",
            transaction: &Transaction{
                Amount:     100,
                Currency:   "USD",
                Type:       "credit_card",
                CardNumber: "4111111111111111",
                CVV:        "123",
            },
            expectError: false,
        },
        {
            name: "invalid credit card - missing number",
            transaction: &Transaction{
                Amount:   100,
                Currency: "USD",
                Type:     "credit_card",
                CVV:      "123",
            },
            expectError: true,
            errorMsg:    "card number required",
        },
        {
            name: "valid bank transfer",
            transaction: &Transaction{
                Amount:        100,
                Currency:      "USD",
                Type:          "bank_transfer",
                AccountNumber: "123456789",
                RoutingNumber: "987654321",
            },
            expectError: false,
        },
        {
            name: "unsupported transaction type",
            transaction: &Transaction{
                Amount:   100,
                Currency: "USD",
                Type:     "cryptocurrency",
            },
            expectError: true,
            errorMsg:    "unsupported transaction type",
        },
    }
    
    for _, tc := range testCases {
        t.Run(tc.name, func(t *testing.T) {
            err := processor.ProcessTransaction(tc.transaction)
            
            if tc.expectError {
                require.Error(t, err)
                assert.Contains(t, err.Error(), tc.errorMsg)
            } else {
                assert.NoError(t, err)
            }
        })
    }
}

// Benchmark test for performance tracking
func BenchmarkTransactionProcessor_ProcessTransaction(b *testing.B) {
    processor := NewTransactionProcessor()
    transaction := &Transaction{
        Amount:     100,
        Currency:   "USD",
        Type:       "credit_card",
        CardNumber: "4111111111111111",
        CVV:        "123",
    }
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = processor.ProcessTransaction(transaction)
    }
}

// Coverage command:
// go test -coverprofile=coverage.out ./...
// go tool cover -html=coverage.out -o coverage.html
// go tool cover -func=coverage.out | grep total
```

---

## 🔄 **Refactoring Strategies**

### **Common Refactoring Patterns**

```go
// Extract Method Refactoring
// ❌ Before: Long method with multiple responsibilities
func ProcessPayment(payment *Payment) error {
    // Validation
    if payment.Amount <= 0 {
        return errors.New("invalid amount")
    }
    if payment.Currency == "" {
        return errors.New("currency required")
    }
    if !isValidCurrency(payment.Currency) {
        return errors.New("unsupported currency")
    }
    
    // Fee calculation
    var fee float64
    if payment.Amount > 1000 {
        fee = payment.Amount * 0.02 // 2% for high value
    } else {
        fee = payment.Amount * 0.03 // 3% for regular
    }
    
    // Apply fee
    payment.TotalAmount = payment.Amount + fee
    
    // Fraud check
    if payment.Amount > 10000 {
        fraudScore := calculateFraudScore(payment.UserID, payment.Amount)
        if fraudScore > 0.8 {
            return errors.New("transaction flagged for fraud")
        }
    }
    
    // Save to database
    db := getDB()
    _, err := db.Exec("INSERT INTO payments...", payment.ID, payment.Amount, payment.TotalAmount)
    return err
}

// ✅ After: Extracted methods with clear responsibilities
type PaymentProcessor struct {
    db             *sql.DB
    fraudDetector  FraudDetector
    feeCalculator  FeeCalculator
    validator      PaymentValidator
}

func (p *PaymentProcessor) ProcessPayment(payment *Payment) error {
    if err := p.validator.ValidatePayment(payment); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }
    
    if err := p.applyFees(payment); err != nil {
        return fmt.Errorf("fee calculation failed: %w", err)
    }
    
    if err := p.checkForFraud(payment); err != nil {
        return fmt.Errorf("fraud check failed: %w", err)
    }
    
    if err := p.savePayment(payment); err != nil {
        return fmt.Errorf("save failed: %w", err)
    }
    
    return nil
}

func (p *PaymentProcessor) applyFees(payment *Payment) error {
    fee, err := p.feeCalculator.Calculate(payment.Amount, payment.Currency)
    if err != nil {
        return err
    }
    
    payment.Fee = fee
    payment.TotalAmount = payment.Amount + fee
    return nil
}

func (p *PaymentProcessor) checkForFraud(payment *Payment) error {
    if payment.Amount <= FraudCheckThreshold {
        return nil
    }
    
    risk, err := p.fraudDetector.AssessRisk(payment)
    if err != nil {
        return fmt.Errorf("fraud assessment failed: %w", err)
    }
    
    if risk.Score > MaxAllowedRiskScore {
        return fmt.Errorf("transaction flagged for fraud: score %.2f", risk.Score)
    }
    
    return nil
}

func (p *PaymentProcessor) savePayment(payment *Payment) error {
    query := `INSERT INTO payments (id, user_id, amount, fee, total_amount, currency, status) 
              VALUES ($1, $2, $3, $4, $5, $6, $7)`
    
    _, err := p.db.Exec(query, payment.ID, payment.UserID, payment.Amount, 
                       payment.Fee, payment.TotalAmount, payment.Currency, payment.Status)
    
    if err != nil {
        return fmt.Errorf("database insert failed: %w", err)
    }
    
    return nil
}

// Replace Conditional with Polymorphism
// ❌ Before: Complex conditional logic
func CalculateShippingCost(order *Order) float64 {
    switch order.ShippingMethod {
    case "standard":
        if order.Weight > 10 {
            return 15.0
        }
        return 10.0
    case "express":
        if order.Weight > 10 {
            return 25.0
        }
        return 20.0
    case "overnight":
        if order.Weight > 10 {
            return 40.0
        }
        return 35.0
    default:
        return 0.0
    }
}

// ✅ After: Polymorphic approach
type ShippingCalculator interface {
    Calculate(weight float64) float64
    GetName() string
}

type StandardShipping struct{}

func (s *StandardShipping) Calculate(weight float64) float64 {
    if weight > 10 {
        return 15.0
    }
    return 10.0
}

func (s *StandardShipping) GetName() string {
    return "standard"
}

type ExpressShipping struct{}

func (e *ExpressShipping) Calculate(weight float64) float64 {
    if weight > 10 {
        return 25.0
    }
    return 20.0
}

func (e *ExpressShipping) GetName() string {
    return "express"
}

type OvernightShipping struct{}

func (o *OvernightShipping) Calculate(weight float64) float64 {
    if weight > 10 {
        return 40.0
    }
    return 35.0
}

func (o *OvernightShipping) GetName() string {
    return "overnight"
}

type ShippingService struct {
    calculators map[string]ShippingCalculator
}

func NewShippingService() *ShippingService {
    return &ShippingService{
        calculators: map[string]ShippingCalculator{
            "standard":  &StandardShipping{},
            "express":   &ExpressShipping{},
            "overnight": &OvernightShipping{},
        },
    }
}

func (s *ShippingService) CalculateShippingCost(order *Order) (float64, error) {
    calculator, exists := s.calculators[order.ShippingMethod]
    if !exists {
        return 0, fmt.Errorf("unknown shipping method: %s", order.ShippingMethod)
    }
    
    return calculator.Calculate(order.Weight), nil
}
```

---

## 💳 **Technical Debt Management**

### **Technical Debt Tracking**

```go
// Technical Debt Documentation Template
type TechnicalDebtItem struct {
    ID          string
    Title       string
    Description string
    Impact      DebtImpact
    Effort      DebtEffort
    CreatedDate time.Time
    Component   string
    Owner       string
    Priority    DebtPriority
    Status      DebtStatus
    Links       []string // Links to issues, PRs, documentation
}

type DebtImpact string

const (
    ImpactLow    DebtImpact = "low"    // Minor inconvenience
    ImpactMedium DebtImpact = "medium" // Affects productivity
    ImpactHigh   DebtImpact = "high"   // Blocks development
    ImpactCritical DebtImpact = "critical" // Risk to business
)

type DebtEffort string

const (
    EffortSmall  DebtEffort = "small"  // < 1 day
    EffortMedium DebtEffort = "medium" // 1-3 days
    EffortLarge  DebtEffort = "large"  // 1-2 weeks
    EffortXLarge DebtEffort = "xlarge" // > 2 weeks
)

// Example of well-documented technical debt
/*
TODO: TECH-DEBT-001 - Refactor payment processing pipeline
Priority: High
Impact: High - Current implementation makes it difficult to add new payment methods
Effort: Large - Requires restructuring core payment logic
Component: payment-service
Owner: backend-team
Created: 2024-01-15
Links: 
  - https://github.com/company/project/issues/123
  - https://docs.company.com/architecture/payment-refactoring

Current state:
- Monolithic payment processor with hardcoded payment methods
- Difficult to test individual payment flows
- Performance issues with complex conditional logic

Desired state:
- Strategy pattern for payment methods
- Separate validation, processing, and notification concerns
- Improved testability and performance

Risks if not addressed:
- Continued difficulty adding new payment methods
- Maintenance overhead
- Performance degradation as complexity grows
*/

// Example of addressing technical debt incrementally
// Step 1: Extract interface (preparation for strategy pattern)
type PaymentMethod interface {
    Process(ctx context.Context, payment *Payment) (*PaymentResult, error)
    Validate(payment *Payment) error
    GetName() string
}

// Step 2: Implement new payment methods using the interface
type CreditCardMethod struct {
    gateway CreditCardGateway
}

func (c *CreditCardMethod) Process(ctx context.Context, payment *Payment) (*PaymentResult, error) {
    return c.gateway.ProcessPayment(ctx, payment)
}

func (c *CreditCardMethod) Validate(payment *Payment) error {
    return validateCreditCard(payment)
}

func (c *CreditCardMethod) GetName() string {
    return "credit_card"
}

// Step 3: Gradually migrate existing code
type PaymentProcessor struct {
    methods map[string]PaymentMethod
    legacy  *LegacyPaymentProcessor // Keep for backward compatibility
}

func (p *PaymentProcessor) ProcessPayment(ctx context.Context, payment *Payment) (*PaymentResult, error) {
    // Check if new method is available
    if method, exists := p.methods[payment.Type]; exists {
        return method.Process(ctx, payment)
    }
    
    // Fall back to legacy processor
    // TODO: TECH-DEBT-001 - Remove this fallback after migration
    return p.legacy.ProcessPayment(ctx, payment)
}
```

### **Debt Prioritization Framework**

```go
// Technical Debt Score Calculator
type DebtScorer struct {
    impactWeights map[DebtImpact]float64
    effortWeights map[DebtEffort]float64
}

func NewDebtScorer() *DebtScorer {
    return &DebtScorer{
        impactWeights: map[DebtImpact]float64{
            ImpactLow:      1.0,
            ImpactMedium:   2.0,
            ImpactHigh:     4.0,
            ImpactCritical: 8.0,
        },
        effortWeights: map[DebtEffort]float64{
            EffortSmall:  1.0,
            EffortMedium: 2.0,
            EffortLarge:  4.0,
            EffortXLarge: 8.0,
        },
    }
}

func (s *DebtScorer) CalculateScore(debt TechnicalDebtItem) float64 {
    impactScore := s.impactWeights[debt.Impact]
    effortScore := s.effortWeights[debt.Effort]
    
    // Higher impact, lower effort = higher priority
    return impactScore / effortScore
}

func (s *DebtScorer) PrioritizeDebts(debts []TechnicalDebtItem) []TechnicalDebtItem {
    // Sort by score descending
    sort.Slice(debts, func(i, j int) bool {
        return s.CalculateScore(debts[i]) > s.CalculateScore(debts[j])
    })
    
    return debts
}

// Debt Budget Allocation
type DebtBudget struct {
    SprintCapacity     int // Story points or hours per sprint
    DebtAllocation     float64 // Percentage allocated to debt (e.g., 0.2 = 20%)
    MinimumAllocation  int // Minimum points/hours for debt
}

func (b *DebtBudget) CalculateDebtCapacity(sprintCapacity int) int {
    allocated := int(float64(sprintCapacity) * b.DebtAllocation)
    if allocated < b.MinimumAllocation {
        return b.MinimumAllocation
    }
    return allocated
}

// Example usage in sprint planning
func PlanDebtWork(debts []TechnicalDebtItem, budget DebtBudget, sprintCapacity int) []TechnicalDebtItem {
    scorer := NewDebtScorer()
    prioritizedDebts := scorer.PrioritizeDebts(debts)
    
    debtCapacity := budget.CalculateDebtCapacity(sprintCapacity)
    plannedDebts := []TechnicalDebtItem{}
    usedCapacity := 0
    
    for _, debt := range prioritizedDebts {
        effortRequired := getEffortInPoints(debt.Effort)
        if usedCapacity + effortRequired <= debtCapacity {
            plannedDebts = append(plannedDebts, debt)
            usedCapacity += effortRequired
        }
    }
    
    return plannedDebts
}

func getEffortInPoints(effort DebtEffort) int {
    switch effort {
    case EffortSmall:
        return 1
    case EffortMedium:
        return 3
    case EffortLarge:
        return 8
    case EffortXLarge:
        return 13
    default:
        return 1
    }
}
```

---

## 🎯 **Interview Questions**

### **Code Quality Questions**

**Q1: How do you maintain code quality in a fast-paced development environment?**

**Answer:**

**Strategies for maintaining code quality under pressure:**

1. **Automated Quality Gates:**
```bash
# CI/CD Pipeline Quality Checks
stages:
  - test
  - lint
  - security-scan
  - code-coverage
  - deploy

lint:
  script:
    - golangci-lint run
    - gofmt -d .
  rules:
    - if: $CI_MERGE_REQUEST_ID

test:
  script:
    - go test -race -coverprofile=coverage.out ./...
    - go tool cover -func=coverage.out | tail -1
  coverage: '/total:\s+\(statements\)\s+(\d+\.\d+\%)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
```

2. **Definition of Done:**
- Code reviewed by at least one other developer
- Unit tests written and passing
- Integration tests passing
- Static analysis checks passing
- Documentation updated
- No critical security vulnerabilities

3. **Technical Debt Management:**
- Allocate 20% of sprint capacity to technical debt
- Track debt items with impact and effort estimates
- Regular architecture reviews

**Q2: How do you handle code reviews for junior developers?**

**Answer:**

**Effective code review strategies:**

1. **Educational Approach:**
```go
// Instead of: "This is wrong"
// Write: "Consider using context.Context here for better timeout handling:
//        func ProcessPayment(ctx context.Context, payment *Payment) error
//        This allows callers to set timeouts and cancel operations."

// Provide examples and explain the reasoning
func ReviewExample() {
    // ❌ Current approach
    result, err := http.Get("https://api.example.com/data")
    
    // ✅ Suggested improvement
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    req, _ := http.NewRequestWithContext(ctx, "GET", "https://api.example.com/data", nil)
    result, err := http.DefaultClient.Do(req)
    
    // Benefits:
    // 1. Prevents hanging requests
    // 2. Allows graceful cancellation
    // 3. Better resource management
}
```

2. **Focus Areas for Junior Developers:**
- Error handling patterns
- Input validation
- Code organization and structure
- Testing practices
- Security considerations

3. **Positive Reinforcement:**
- Acknowledge good practices
- Explain why certain approaches are preferred
- Share learning resources

**Q3: Describe your approach to refactoring legacy code.**

**Answer:**

**Safe refactoring strategy:**

1. **Assessment Phase:**
```go
// Before refactoring, understand the current system
type LegacyAnalysis struct {
    ComplexityScore     int
    TestCoverage       float64
    Dependencies       []string
    KnownIssues        []string
    BusinessCriticality string
}

func AnalyzeLegacyCode(codebase string) LegacyAnalysis {
    // Use tools like:
    // - gocyclo for complexity analysis
    // - go test -cover for coverage
    // - dependency analysis tools
    // - static analysis results
    
    return LegacyAnalysis{
        ComplexityScore: 15, // High complexity
        TestCoverage:   0.3, // Low coverage
        Dependencies:   []string{"database", "external-api", "cache"},
        KnownIssues:    []string{"memory leaks", "race conditions"},
        BusinessCriticality: "high",
    }
}
```

2. **Incremental Refactoring:**
```go
// Step 1: Add characterization tests
func TestLegacyBehavior(t *testing.T) {
    // Test current behavior to prevent regressions
    input := createTestInput()
    output := legacyFunction(input)
    
    // Document current behavior, even if not ideal
    assert.Equal(t, expectedCurrentOutput, output)
}

// Step 2: Extract interfaces
type PaymentProcessor interface {
    Process(payment *Payment) error
}

// Step 3: Create new implementation alongside old
type ModernPaymentProcessor struct {
    validator PaymentValidator
    gateway   PaymentGateway
}

// Step 4: Use adapter pattern for gradual migration
type PaymentProcessorAdapter struct {
    modern PaymentProcessor
    legacy *LegacyPaymentProcessor
    useModern bool
}

func (a *PaymentProcessorAdapter) Process(payment *Payment) error {
    if a.useModern {
        return a.modern.Process(payment)
    }
    return a.legacy.Process(payment)
}
```

3. **Risk Management:**
- Feature flags for gradual rollout
- Comprehensive monitoring
- Rollback plans
- Parallel processing for validation

**Q4: How do you measure and improve code quality metrics?**

**Answer:**

**Key metrics and improvement strategies:**

1. **Quantitative Metrics:**
```go
type QualityMetrics struct {
    CodeCoverage        float64 // Target: >80%
    CyclomaticComplexity int    // Target: <10 per function
    CodeDuplication     float64 // Target: <3%
    TechnicalDebtRatio  float64 // Target: <5%
    DefectDensity       float64 // Defects per KLOC
    MaintainabilityIndex float64 // 0-100 scale
}

// Automated metric collection
func CollectMetrics(codebase string) QualityMetrics {
    coverage := runCoverageAnalysis(codebase)
    complexity := runComplexityAnalysis(codebase)
    duplication := runDuplicationAnalysis(codebase)
    debt := calculateTechnicalDebt(codebase)
    
    return QualityMetrics{
        CodeCoverage:        coverage,
        CyclomaticComplexity: complexity,
        CodeDuplication:     duplication,
        TechnicalDebtRatio:  debt,
    }
}
```

2. **Improvement Strategies:**
- Set quality gates in CI/CD
- Regular code quality reviews
- Developer training and guidelines
- Automated refactoring tools
- Debt tracking and prioritization

3. **Monitoring and Reporting:**
- Dashboard with trending metrics
- Alerts for quality regressions
- Team retrospectives on quality issues
- Celebrate quality improvements

This comprehensive code quality guide provides the framework for maintaining high standards in software development and demonstrates the expertise expected from senior engineers in technical interviews.

## Performance Optimization

<!-- AUTO-GENERATED ANCHOR: originally referenced as #performance-optimization -->

Placeholder content. Please replace with proper section.


## Team Practices  Culture

<!-- AUTO-GENERATED ANCHOR: originally referenced as #team-practices--culture -->

Placeholder content. Please replace with proper section.
