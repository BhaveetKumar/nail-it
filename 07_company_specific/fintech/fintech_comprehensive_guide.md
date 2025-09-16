# 💰 Fintech Comprehensive Guide for Backend Engineers

## Table of Contents
1. [Payment Systems](#payment-systems/)
2. [Banking Infrastructure](#banking-infrastructure/)
3. [Cryptocurrency & Blockchain](#cryptocurrency--blockchain/)
4. [Risk Management](#risk-management/)
5. [Regulatory Compliance](#regulatory-compliance/)
6. [Security in Fintech](#security-in-fintech/)
7. [Real-time Processing](#real-time-processing/)
8. [Data Analytics](#data-analytics/)
9. [System Design Patterns](#system-design-patterns/)
10. [Interview Questions](#interview-questions/)

## Payment Systems

### Payment Gateway Architecture

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

type PaymentRequest struct {
    Amount      int64  `json:"amount"`
    Currency    string `json:"currency"`
    CardNumber  string `json:"card_number"`
    CVV         string `json:"cvv"`
    ExpiryMonth int    `json:"expiry_month"`
    ExpiryYear  int    `json:"expiry_year"`
    MerchantID  string `json:"merchant_id"`
    OrderID     string `json:"order_id"`
}

type PaymentResponse struct {
    TransactionID string    `json:"transaction_id"`
    Status        string    `json:"status"`
    Amount        int64     `json:"amount"`
    Currency      string    `json:"currency"`
    Timestamp     time.Time `json:"timestamp"`
    ErrorCode     string    `json:"error_code,omitempty"`
    ErrorMessage  string    `json:"error_message,omitempty"`
}

type PaymentGateway struct {
    acquirerURL string
    apiKey      string
    client      *http.Client
}

func NewPaymentGateway(acquirerURL, apiKey string) *PaymentGateway {
    return &PaymentGateway{
        acquirerURL: acquirerURL,
        apiKey:      apiKey,
        client:      &http.Client{Timeout: 30 * time.Second},
    }
}

func (pg *PaymentGateway) ProcessPayment(req *PaymentRequest) (*PaymentResponse, error) {
    // Validate payment request
    if err := pg.validatePayment(req); err != nil {
        return &PaymentResponse{
            Status:       "failed",
            ErrorCode:    "VALIDATION_ERROR",
            ErrorMessage: err.Error(),
        }, nil
    }
    
    // Tokenize card details
    token, err := pg.tokenizeCard(req)
    if err != nil {
        return &PaymentResponse{
            Status:       "failed",
            ErrorCode:    "TOKENIZATION_ERROR",
            ErrorMessage: err.Error(),
        }, nil
    }
    
    // Process payment with acquirer
    response, err := pg.sendToAcquirer(req, token)
    if err != nil {
        return &PaymentResponse{
            Status:       "failed",
            ErrorCode:    "ACQUIRER_ERROR",
            ErrorMessage: err.Error(),
        }, nil
    }
    
    return response, nil
}

func (pg *PaymentGateway) validatePayment(req *PaymentRequest) error {
    if req.Amount <= 0 {
        return fmt.Errorf("invalid amount")
    }
    if len(req.CardNumber) < 13 || len(req.CardNumber) > 19 {
        return fmt.Errorf("invalid card number")
    }
    if len(req.CVV) < 3 || len(req.CVV) > 4 {
        return fmt.Errorf("invalid CVV")
    }
    return nil
}

func (pg *PaymentGateway) tokenizeCard(req *PaymentRequest) (string, error) {
    // In real implementation, this would call a tokenization service
    // For demo purposes, we'll generate a mock token
    return fmt.Sprintf("tok_%s_%d", req.CardNumber[len(req.CardNumber)-4:], time.Now().Unix()), nil
}

func (pg *PaymentGateway) sendToAcquirer(req *PaymentRequest, token string) (*PaymentResponse, error) {
    // Mock acquirer response
    return &PaymentResponse{
        TransactionID: fmt.Sprintf("txn_%d", time.Now().Unix()),
        Status:        "success",
        Amount:        req.Amount,
        Currency:      req.Currency,
        Timestamp:     time.Now(),
    }, nil
}

func main() {
    gateway := NewPaymentGateway("https://api.acquirer.com", "sk_test_123")
    
    req := &PaymentRequest{
        Amount:      10000, // $100.00 in cents
        Currency:    "USD",
        CardNumber:  "4111111111111111",
        CVV:         "123",
        ExpiryMonth: 12,
        ExpiryYear:  2025,
        MerchantID:  "merchant_123",
        OrderID:     "order_456",
    }
    
    resp, err := gateway.ProcessPayment(req)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    fmt.Printf("Payment Response: %+v\n", resp)
}
```

### Payment Processing Flow

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type PaymentProcessor struct {
    validationQueue chan *PaymentRequest
    processingQueue chan *PaymentRequest
    resultQueue     chan *PaymentResponse
    workers         int
}

func NewPaymentProcessor(workers int) *PaymentProcessor {
    return &PaymentProcessor{
        validationQueue: make(chan *PaymentRequest, 1000),
        processingQueue: make(chan *PaymentRequest, 1000),
        resultQueue:     make(chan *PaymentResponse, 1000),
        workers:         workers,
    }
}

func (pp *PaymentProcessor) Start(ctx context.Context) {
    var wg sync.WaitGroup
    
    // Start validation workers
    for i := 0; i < pp.workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            pp.validationWorker(ctx)
        }()
    }
    
    // Start processing workers
    for i := 0; i < pp.workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            pp.processingWorker(ctx)
        }()
    }
    
    wg.Wait()
}

func (pp *PaymentProcessor) validationWorker(ctx context.Context) {
    for {
        select {
        case req := <-pp.validationQueue:
            if pp.validatePayment(req) {
                pp.processingQueue <- req
            } else {
                pp.resultQueue <- &PaymentResponse{
                    Status:       "failed",
                    ErrorCode:    "VALIDATION_ERROR",
                    ErrorMessage: "Payment validation failed",
                }
            }
        case <-ctx.Done():
            return
        }
    }
}

func (pp *PaymentProcessor) processingWorker(ctx context.Context) {
    for {
        select {
        case req := <-pp.processingQueue:
            // Simulate payment processing
            time.Sleep(100 * time.Millisecond)
            
            pp.resultQueue <- &PaymentResponse{
                TransactionID: fmt.Sprintf("txn_%d", time.Now().UnixNano()),
                Status:        "success",
                Amount:        req.Amount,
                Currency:      req.Currency,
                Timestamp:     time.Now(),
            }
        case <-ctx.Done():
            return
        }
    }
}

func (pp *PaymentProcessor) validatePayment(req *PaymentRequest) bool {
    // Basic validation logic
    return req.Amount > 0 && len(req.CardNumber) >= 13
}

func (pp *PaymentProcessor) SubmitPayment(req *PaymentRequest) {
    pp.validationQueue <- req
}

func (pp *PaymentProcessor) GetResult() <-chan *PaymentResponse {
    return pp.resultQueue
}
```

## Banking Infrastructure

### Account Management System

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Account struct {
    ID          string    `json:"id"`
    UserID      string    `json:"user_id"`
    AccountType string    `json:"account_type"`
    Balance     int64     `json:"balance"` // in cents
    Currency    string    `json:"currency"`
    CreatedAt   time.Time `json:"created_at"`
    UpdatedAt   time.Time `json:"updated_at"`
}

type Transaction struct {
    ID              string    `json:"id"`
    FromAccountID   string    `json:"from_account_id"`
    ToAccountID     string    `json:"to_account_id"`
    Amount          int64     `json:"amount"`
    Currency        string    `json:"currency"`
    TransactionType string    `json:"transaction_type"`
    Status          string    `json:"status"`
    Description     string    `json:"description"`
    CreatedAt       time.Time `json:"created_at"`
}

type AccountService struct {
    accounts     map[string]*Account
    transactions map[string]*Transaction
    mutex        sync.RWMutex
}

func NewAccountService() *AccountService {
    return &AccountService{
        accounts:     make(map[string]*Account),
        transactions: make(map[string]*Transaction),
    }
}

func (as *AccountService) CreateAccount(userID, accountType, currency string) (*Account, error) {
    as.mutex.Lock()
    defer as.mutex.Unlock()
    
    accountID := fmt.Sprintf("acc_%d", time.Now().UnixNano())
    account := &Account{
        ID:          accountID,
        UserID:      userID,
        AccountType: accountType,
        Balance:     0,
        Currency:    currency,
        CreatedAt:   time.Now(),
        UpdatedAt:   time.Now(),
    }
    
    as.accounts[accountID] = account
    return account, nil
}

func (as *AccountService) GetAccount(accountID string) (*Account, error) {
    as.mutex.RLock()
    defer as.mutex.RUnlock()
    
    account, exists := as.accounts[accountID]
    if !exists {
        return nil, fmt.Errorf("account not found")
    }
    
    return account, nil
}

func (as *AccountService) Transfer(fromAccountID, toAccountID string, amount int64, description string) (*Transaction, error) {
    as.mutex.Lock()
    defer as.mutex.Unlock()
    
    // Validate accounts
    fromAccount, exists := as.accounts[fromAccountID]
    if !exists {
        return nil, fmt.Errorf("from account not found")
    }
    
    toAccount, exists := as.accounts[toAccountID]
    if !exists {
        return nil, fmt.Errorf("to account not found")
    }
    
    // Check sufficient balance
    if fromAccount.Balance < amount {
        return nil, fmt.Errorf("insufficient balance")
    }
    
    // Create transaction
    transactionID := fmt.Sprintf("txn_%d", time.Now().UnixNano())
    transaction := &Transaction{
        ID:              transactionID,
        FromAccountID:   fromAccountID,
        ToAccountID:     toAccountID,
        Amount:          amount,
        Currency:        fromAccount.Currency,
        TransactionType: "transfer",
        Status:          "pending",
        Description:     description,
        CreatedAt:       time.Now(),
    }
    
    // Update balances
    fromAccount.Balance -= amount
    toAccount.Balance += amount
    fromAccount.UpdatedAt = time.Now()
    toAccount.UpdatedAt = time.Now()
    
    // Update transaction status
    transaction.Status = "completed"
    
    as.transactions[transactionID] = transaction
    
    return transaction, nil
}

func (as *AccountService) GetTransactionHistory(accountID string) ([]*Transaction, error) {
    as.mutex.RLock()
    defer as.mutex.RUnlock()
    
    var history []*Transaction
    for _, transaction := range as.transactions {
        if transaction.FromAccountID == accountID || transaction.ToAccountID == accountID {
            history = append(history, transaction)
        }
    }
    
    return history, nil
}
```

### Interest Calculation Engine

```go
package main

import (
    "fmt"
    "math"
    "time"
)

type InterestRate struct {
    AccountType string    `json:"account_type"`
    Rate        float64   `json:"rate"` // Annual percentage rate
    EffectiveFrom time.Time `json:"effective_from"`
    EffectiveTo   time.Time `json:"effective_to"`
}

type InterestCalculator struct {
    rates []InterestRate
}

func NewInterestCalculator() *InterestCalculator {
    return &InterestCalculator{
        rates: []InterestRate{
            {
                AccountType:  "savings",
                Rate:         2.5, // 2.5% APR
                EffectiveFrom: time.Now().AddDate(-1, 0, 0),
                EffectiveTo:   time.Now().AddDate(1, 0, 0),
            },
            {
                AccountType:  "checking",
                Rate:         0.1, // 0.1% APR
                EffectiveFrom: time.Now().AddDate(-1, 0, 0),
                EffectiveTo:   time.Now().AddDate(1, 0, 0),
            },
        },
    }
}

func (ic *InterestCalculator) CalculateInterest(account *Account, fromDate, toDate time.Time) (int64, error) {
    // Find applicable interest rate
    rate := ic.getApplicableRate(account.AccountType, fromDate, toDate)
    if rate == 0 {
        return 0, nil
    }
    
    // Calculate daily interest rate
    dailyRate := rate / 365.0 / 100.0
    
    // Calculate number of days
    days := int(toDate.Sub(fromDate).Hours() / 24)
    
    // Calculate compound interest
    // A = P(1 + r/n)^(nt)
    // For daily compounding: n = 365
    principal := float64(account.Balance)
    compoundFactor := math.Pow(1+dailyRate, float64(days))
    finalAmount := principal * compoundFactor
    
    interest := int64(finalAmount - principal)
    return interest, nil
}

func (ic *InterestCalculator) getApplicableRate(accountType string, fromDate, toDate time.Time) float64 {
    for _, rate := range ic.rates {
        if rate.AccountType == accountType &&
            rate.EffectiveFrom.Before(toDate) &&
            rate.EffectiveTo.After(fromDate) {
            return rate.Rate
        }
    }
    return 0
}

func (ic *InterestCalculator) CalculateCompoundInterest(principal int64, rate float64, timeInYears float64, compoundingFrequency int) int64 {
    // A = P(1 + r/n)^(nt)
    // n = compounding frequency per year
    // t = time in years
    
    principalFloat := float64(principal)
    rateDecimal := rate / 100.0
    nt := float64(compoundingFrequency) * timeInYears
    
    amount := principalFloat * math.Pow(1+rateDecimal/float64(compoundingFrequency), nt)
    return int64(amount)
}
```

## Cryptocurrency & Blockchain

### Basic Blockchain Implementation

```go
package main

import (
    "crypto/sha256"
    "encoding/hex"
    "fmt"
    "strconv"
    "time"
)

type Block struct {
    Index        int           `json:"index"`
    Timestamp    int64         `json:"timestamp"`
    Data         string        `json:"data"`
    PreviousHash string        `json:"previous_hash"`
    Hash         string        `json:"hash"`
    Nonce        int           `json:"nonce"`
}

type Blockchain struct {
    chain []Block
}

func NewBlockchain() *Blockchain {
    genesisBlock := Block{
        Index:        0,
        Timestamp:    time.Now().Unix(),
        Data:         "Genesis Block",
        PreviousHash: "0",
        Hash:         "",
        Nonce:        0,
    }
    
    genesisBlock.Hash = calculateHash(genesisBlock)
    
    return &Blockchain{
        chain: []Block{genesisBlock},
    }
}

func calculateHash(block Block) string {
    record := strconv.Itoa(block.Index) + 
              strconv.FormatInt(block.Timestamp, 10) + 
              block.Data + 
              block.PreviousHash + 
              strconv.Itoa(block.Nonce)
    
    hash := sha256.Sum256([]byte(record))
    return hex.EncodeToString(hash[:])
}

func (bc *Blockchain) AddBlock(data string) {
    previousBlock := bc.chain[len(bc.chain)-1]
    newBlock := Block{
        Index:        previousBlock.Index + 1,
        Timestamp:    time.Now().Unix(),
        Data:         data,
        PreviousHash: previousBlock.Hash,
        Hash:         "",
        Nonce:        0,
    }
    
    // Proof of Work
    newBlock = proofOfWork(newBlock)
    
    bc.chain = append(bc.chain, newBlock)
}

func proofOfWork(block Block) Block {
    target := "0000" // Difficulty target
    
    for {
        hash := calculateHash(block)
        if hash[:len(target)] == target {
            block.Hash = hash
            break
        }
        block.Nonce++
    }
    
    return block
}

func (bc *Blockchain) IsValid() bool {
    for i := 1; i < len(bc.chain); i++ {
        currentBlock := bc.chain[i]
        previousBlock := bc.chain[i-1]
        
        // Check if current block hash is correct
        if currentBlock.Hash != calculateHash(currentBlock) {
            return false
        }
        
        // Check if current block points to previous block
        if currentBlock.PreviousHash != previousBlock.Hash {
            return false
        }
    }
    
    return true
}

func (bc *Blockchain) PrintChain() {
    for _, block := range bc.chain {
        fmt.Printf("Block %d:\n", block.Index)
        fmt.Printf("  Timestamp: %d\n", block.Timestamp)
        fmt.Printf("  Data: %s\n", block.Data)
        fmt.Printf("  Previous Hash: %s\n", block.PreviousHash)
        fmt.Printf("  Hash: %s\n", block.Hash)
        fmt.Printf("  Nonce: %d\n", block.Nonce)
        fmt.Println()
    }
}
```

### Cryptocurrency Wallet

```go
package main

import (
    "crypto/ecdsa"
    "crypto/elliptic"
    "crypto/rand"
    "crypto/sha256"
    "encoding/hex"
    "fmt"
    "math/big"
)

type Wallet struct {
    PrivateKey *ecdsa.PrivateKey
    PublicKey  []byte
    Address    string
}

func NewWallet() *Wallet {
    privateKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
    publicKey := append(privateKey.PublicKey.X.Bytes(), privateKey.PublicKey.Y.Bytes()...)
    
    wallet := &Wallet{
        PrivateKey: privateKey,
        PublicKey:  publicKey,
    }
    
    wallet.Address = wallet.generateAddress()
    return wallet
}

func (w *Wallet) generateAddress() string {
    publicKeyHash := sha256.Sum256(w.PublicKey)
    return hex.EncodeToString(publicKeyHash[:])
}

func (w *Wallet) Sign(data []byte) ([]byte, error) {
    hash := sha256.Sum256(data)
    r, s, err := ecdsa.Sign(rand.Reader, w.PrivateKey, hash[:])
    if err != nil {
        return nil, err
    }
    
    signature := append(r.Bytes(), s.Bytes()...)
    return signature, nil
}

func (w *Wallet) VerifySignature(data []byte, signature []byte) bool {
    hash := sha256.Sum256(data)
    
    r := new(big.Int).SetBytes(signature[:len(signature)/2])
    s := new(big.Int).SetBytes(signature[len(signature)/2:])
    
    return ecdsa.Verify(&w.PrivateKey.PublicKey, hash[:], r, s)
}

type Transaction struct {
    From   string `json:"from"`
    To     string `json:"to"`
    Amount int64  `json:"amount"`
    Data   []byte `json:"data"`
    Signature []byte `json:"signature"`
}

func (w *Wallet) CreateTransaction(to string, amount int64) *Transaction {
    data := []byte(fmt.Sprintf("%s%s%d", w.Address, to, amount))
    signature, _ := w.Sign(data)
    
    return &Transaction{
        From:      w.Address,
        To:        to,
        Amount:    amount,
        Data:      data,
        Signature: signature,
    }
}

func (tx *Transaction) VerifyTransaction(wallet *Wallet) bool {
    data := []byte(fmt.Sprintf("%s%s%d", tx.From, tx.To, tx.Amount))
    return wallet.VerifySignature(data, tx.Signature)
}
```

## Risk Management

### Fraud Detection System

```go
package main

import (
    "fmt"
    "math"
    "time"
)

type Transaction struct {
    ID          string    `json:"id"`
    UserID      string    `json:"user_id"`
    Amount      int64     `json:"amount"`
    MerchantID  string    `json:"merchant_id"`
    Timestamp   time.Time `json:"timestamp"`
    Location    string    `json:"location"`
    DeviceID    string    `json:"device_id"`
    IPAddress   string    `json:"ip_address"`
}

type RiskScore struct {
    TransactionID string  `json:"transaction_id"`
    Score         float64 `json:"score"`
    RiskLevel     string  `json:"risk_level"`
    Reasons       []string `json:"reasons"`
}

type FraudDetector struct {
    rules []FraudRule
}

type FraudRule interface {
    Evaluate(tx *Transaction, history []*Transaction) (float64, []string)
}

type AmountRule struct {
    threshold int64
    weight    float64
}

func (r *AmountRule) Evaluate(tx *Transaction, history []*Transaction) (float64, []string) {
    if tx.Amount > r.threshold {
        return r.weight, []string{fmt.Sprintf("Amount %d exceeds threshold %d", tx.Amount, r.threshold)}
    }
    return 0, nil
}

type VelocityRule struct {
    timeWindow time.Duration
    maxCount   int
    weight     float64
}

func (r *VelocityRule) Evaluate(tx *Transaction, history []*Transaction) (float64, []string) {
    cutoff := tx.Timestamp.Add(-r.timeWindow)
    count := 0
    
    for _, htx := range history {
        if htx.UserID == tx.UserID && htx.Timestamp.After(cutoff) {
            count++
        }
    }
    
    if count >= r.maxCount {
        return r.weight, []string{fmt.Sprintf("High velocity: %d transactions in %v", count, r.timeWindow)}
    }
    return 0, nil
}

type LocationRule struct {
    weight float64
}

func (r *LocationRule) Evaluate(tx *Transaction, history []*Transaction) (float64, []string) {
    if len(history) == 0 {
        return 0, nil
    }
    
    // Check if location is different from recent transactions
    recentLocation := history[len(history)-1].Location
    if tx.Location != recentLocation {
        return r.weight, []string{fmt.Sprintf("Location changed from %s to %s", recentLocation, tx.Location)}
    }
    return 0, nil
}

func NewFraudDetector() *FraudDetector {
    return &FraudDetector{
        rules: []FraudRule{
            &AmountRule{threshold: 100000, weight: 0.3}, // $1000
            &VelocityRule{timeWindow: 1 * time.Hour, maxCount: 5, weight: 0.4},
            &LocationRule{weight: 0.3},
        },
    }
}

func (fd *FraudDetector) EvaluateTransaction(tx *Transaction, history []*Transaction) *RiskScore {
    totalScore := 0.0
    var reasons []string
    
    for _, rule := range fd.rules {
        score, ruleReasons := rule.Evaluate(tx, history)
        totalScore += score
        reasons = append(reasons, ruleReasons...)
    }
    
    riskLevel := "low"
    if totalScore > 0.7 {
        riskLevel = "high"
    } else if totalScore > 0.3 {
        riskLevel = "medium"
    }
    
    return &RiskScore{
        TransactionID: tx.ID,
        Score:         totalScore,
        RiskLevel:     riskLevel,
        Reasons:       reasons,
    }
}
```

### Credit Scoring System

```go
package main

import (
    "fmt"
    "math"
    "time"
)

type CreditProfile struct {
    UserID           string    `json:"user_id"`
    CreditScore      int       `json:"credit_score"`
    PaymentHistory   []Payment `json:"payment_history"`
    CreditUtilization float64   `json:"credit_utilization"`
    CreditAge        int       `json:"credit_age_months"`
    RecentInquiries  int       `json:"recent_inquiries"`
    DebtToIncome     float64   `json:"debt_to_income_ratio"`
}

type Payment struct {
    Amount      int64     `json:"amount"`
    DueDate     time.Time `json:"due_date"`
    PaidDate    time.Time `json:"paid_date"`
    IsOnTime    bool      `json:"is_on_time"`
    IsLate      bool      `json:"is_late"`
    DaysLate    int       `json:"days_late"`
}

type CreditScorer struct {
    weights map[string]float64
}

func NewCreditScorer() *CreditScorer {
    return &CreditScorer{
        weights: map[string]float64{
            "payment_history":    0.35,
            "credit_utilization": 0.30,
            "credit_age":         0.15,
            "recent_inquiries":   0.10,
            "debt_to_income":     0.10,
        },
    }
}

func (cs *CreditScorer) CalculateCreditScore(profile *CreditProfile) int {
    // Payment history score (0-100)
    paymentScore := cs.calculatePaymentHistoryScore(profile.PaymentHistory)
    
    // Credit utilization score (0-100)
    utilizationScore := cs.calculateUtilizationScore(profile.CreditUtilization)
    
    // Credit age score (0-100)
    ageScore := cs.calculateAgeScore(profile.CreditAge)
    
    // Recent inquiries score (0-100)
    inquiryScore := cs.calculateInquiryScore(profile.RecentInquiries)
    
    // Debt-to-income score (0-100)
    dtiScore := cs.calculateDTIScore(profile.DebtToIncome)
    
    // Weighted average
    totalScore := paymentScore*cs.weights["payment_history"] +
                  utilizationScore*cs.weights["credit_utilization"] +
                  ageScore*cs.weights["credit_age"] +
                  inquiryScore*cs.weights["recent_inquiries"] +
                  dtiScore*cs.weights["debt_to_income"]
    
    return int(math.Round(totalScore))
}

func (cs *CreditScorer) calculatePaymentHistoryScore(payments []Payment) float64 {
    if len(payments) == 0 {
        return 50 // Neutral score for no history
    }
    
    onTimeCount := 0
    lateCount := 0
    veryLateCount := 0
    
    for _, payment := range payments {
        if payment.IsOnTime {
            onTimeCount++
        } else if payment.IsLate {
            if payment.DaysLate <= 30 {
                lateCount++
            } else {
                veryLateCount++
            }
        }
    }
    
    total := len(payments)
    onTimeRatio := float64(onTimeCount) / float64(total)
    lateRatio := float64(lateCount) / float64(total)
    veryLateRatio := float64(veryLateCount) / float64(total)
    
    // Calculate score based on ratios
    score := onTimeRatio*100 - lateRatio*50 - veryLateRatio*100
    return math.Max(0, math.Min(100, score))
}

func (cs *CreditScorer) calculateUtilizationScore(utilization float64) float64 {
    if utilization <= 0.1 {
        return 100
    } else if utilization <= 0.3 {
        return 90
    } else if utilization <= 0.5 {
        return 70
    } else if utilization <= 0.7 {
        return 50
    } else if utilization <= 0.9 {
        return 30
    } else {
        return 10
    }
}

func (cs *CreditScorer) calculateAgeScore(ageMonths int) float64 {
    if ageMonths >= 84 { // 7+ years
        return 100
    } else if ageMonths >= 60 { // 5+ years
        return 90
    } else if ageMonths >= 36 { // 3+ years
        return 70
    } else if ageMonths >= 24 { // 2+ years
        return 50
    } else if ageMonths >= 12 { // 1+ year
        return 30
    } else {
        return 10
    }
}

func (cs *CreditScorer) calculateInquiryScore(inquiries int) float64 {
    if inquiries == 0 {
        return 100
    } else if inquiries <= 2 {
        return 80
    } else if inquiries <= 4 {
        return 60
    } else if inquiries <= 6 {
        return 40
    } else {
        return 20
    }
}

func (cs *CreditScorer) calculateDTIScore(dti float64) float64 {
    if dti <= 0.2 {
        return 100
    } else if dti <= 0.3 {
        return 90
    } else if dti <= 0.4 {
        return 70
    } else if dti <= 0.5 {
        return 50
    } else if dti <= 0.6 {
        return 30
    } else {
        return 10
    }
}
```

## Regulatory Compliance

### KYC (Know Your Customer) System

```go
package main

import (
    "fmt"
    "regexp"
    "strings"
    "time"
)

type Customer struct {
    ID           string    `json:"id"`
    FirstName    string    `json:"first_name"`
    LastName     string    `json:"last_name"`
    Email        string    `json:"email"`
    Phone        string    `json:"phone"`
    DateOfBirth  time.Time `json:"date_of_birth"`
    Address      Address   `json:"address"`
    DocumentType string    `json:"document_type"`
    DocumentID   string    `json:"document_id"`
    KYCStatus    string    `json:"kyc_status"`
    CreatedAt    time.Time `json:"created_at"`
    UpdatedAt    time.Time `json:"updated_at"`
}

type Address struct {
    Street  string `json:"street"`
    City    string `json:"city"`
    State   string `json:"state"`
    Country string `json:"country"`
    ZIPCode string `json:"zip_code"`
}

type KYCDocument struct {
    ID           string    `json:"id"`
    CustomerID   string    `json:"customer_id"`
    DocumentType string    `json:"document_type"`
    DocumentData []byte    `json:"document_data"`
    Status       string    `json:"status"`
    UploadedAt   time.Time `json:"uploaded_at"`
    VerifiedAt   time.Time `json:"verified_at"`
}

type KYCService struct {
    customers map[string]*Customer
    documents map[string]*KYCDocument
}

func NewKYCService() *KYCService {
    return &KYCService{
        customers: make(map[string]*Customer),
        documents: make(map[string]*KYCDocument),
    }
}

func (ks *KYCService) RegisterCustomer(customer *Customer) error {
    // Validate customer data
    if err := ks.validateCustomer(customer); err != nil {
        return err
    }
    
    // Check for existing customer
    if ks.customerExists(customer.Email, customer.Phone) {
        return fmt.Errorf("customer already exists")
    }
    
    // Generate customer ID
    customer.ID = fmt.Sprintf("cust_%d", time.Now().UnixNano())
    customer.KYCStatus = "pending"
    customer.CreatedAt = time.Now()
    customer.UpdatedAt = time.Now()
    
    ks.customers[customer.ID] = customer
    return nil
}

func (ks *KYCService) validateCustomer(customer *Customer) error {
    // Validate email
    if !ks.isValidEmail(customer.Email) {
        return fmt.Errorf("invalid email format")
    }
    
    // Validate phone
    if !ks.isValidPhone(customer.Phone) {
        return fmt.Errorf("invalid phone format")
    }
    
    // Validate age (must be 18+)
    age := time.Since(customer.DateOfBirth).Hours() / 24 / 365
    if age < 18 {
        return fmt.Errorf("customer must be 18 or older")
    }
    
    // Validate required fields
    if customer.FirstName == "" || customer.LastName == "" {
        return fmt.Errorf("first name and last name are required")
    }
    
    return nil
}

func (ks *KYCService) isValidEmail(email string) bool {
    pattern := `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
    matched, _ := regexp.MatchString(pattern, email)
    return matched
}

func (ks *KYCService) isValidPhone(phone string) bool {
    // Remove all non-digit characters
    digits := regexp.MustCompile(`\D`).ReplaceAllString(phone, "")
    return len(digits) >= 10 && len(digits) <= 15
}

func (ks *KYCService) customerExists(email, phone string) bool {
    for _, customer := range ks.customers {
        if customer.Email == email || customer.Phone == phone {
            return true
        }
    }
    return false
}

func (ks *KYCService) UploadDocument(customerID string, docType string, docData []byte) error {
    customer, exists := ks.customers[customerID]
    if !exists {
        return fmt.Errorf("customer not found")
    }
    
    document := &KYCDocument{
        ID:           fmt.Sprintf("doc_%d", time.Now().UnixNano()),
        CustomerID:   customerID,
        DocumentType: docType,
        DocumentData: docData,
        Status:       "pending",
        UploadedAt:   time.Now(),
    }
    
    ks.documents[document.ID] = document
    
    // Update customer KYC status
    customer.KYCStatus = "document_uploaded"
    customer.UpdatedAt = time.Now()
    
    return nil
}

func (ks *KYCService) VerifyDocument(docID string, verified bool) error {
    document, exists := ks.documents[docID]
    if !exists {
        return fmt.Errorf("document not found")
    }
    
    if verified {
        document.Status = "verified"
        document.VerifiedAt = time.Now()
        
        // Update customer KYC status
        customer := ks.customers[document.CustomerID]
        customer.KYCStatus = "verified"
        customer.UpdatedAt = time.Now()
    } else {
        document.Status = "rejected"
        
        // Update customer KYC status
        customer := ks.customers[document.CustomerID]
        customer.KYCStatus = "rejected"
        customer.UpdatedAt = time.Now()
    }
    
    return nil
}

func (ks *KYCService) GetCustomerKYCStatus(customerID string) (string, error) {
    customer, exists := ks.customers[customerID]
    if !exists {
        return "", fmt.Errorf("customer not found")
    }
    
    return customer.KYCStatus, nil
}
```

## Security in Fintech

### Encryption and Hashing

```go
package main

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "crypto/sha256"
    "encoding/base64"
    "fmt"
    "io"
)

type SecurityService struct {
    encryptionKey []byte
}

func NewSecurityService(key []byte) *SecurityService {
    return &SecurityService{encryptionKey: key}
}

func (ss *SecurityService) Encrypt(plaintext string) (string, error) {
    block, err := aes.NewCipher(ss.encryptionKey)
    if err != nil {
        return "", err
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return "", err
    }
    
    nonce := make([]byte, gcm.NonceSize())
    if _, err = io.ReadFull(rand.Reader, nonce); err != nil {
        return "", err
    }
    
    ciphertext := gcm.Seal(nonce, nonce, []byte(plaintext), nil)
    return base64.StdEncoding.EncodeToString(ciphertext), nil
}

func (ss *SecurityService) Decrypt(ciphertext string) (string, error) {
    data, err := base64.StdEncoding.DecodeString(ciphertext)
    if err != nil {
        return "", err
    }
    
    block, err := aes.NewCipher(ss.encryptionKey)
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

func (ss *SecurityService) HashPassword(password string) string {
    hash := sha256.Sum256([]byte(password))
    return fmt.Sprintf("%x", hash)
}

func (ss *SecurityService) VerifyPassword(password, hash string) bool {
    return ss.HashPassword(password) == hash
}

func (ss *SecurityService) GenerateAPIKey() string {
    bytes := make([]byte, 32)
    if _, err := rand.Read(bytes); err != nil {
        return ""
    }
    return base64.StdEncoding.EncodeToString(bytes)
}
```

## Interview Questions

### Basic Concepts
1. **What is the difference between a payment gateway and a payment processor?**
2. **Explain the payment processing flow.**
3. **What are the key components of a banking system?**
4. **How do you ensure data security in financial applications?**
5. **What is PCI DSS compliance?**

### Advanced Topics
1. **How would you design a real-time fraud detection system?**
2. **Explain the concept of double-entry bookkeeping in software.**
3. **How would you implement a distributed ledger system?**
4. **What are the challenges in building a cryptocurrency exchange?**
5. **How would you design a credit scoring system?**

### System Design
1. **Design a payment processing system for a fintech startup.**
2. **How would you build a peer-to-peer payment system?**
3. **Design a cryptocurrency wallet system.**
4. **How would you implement a real-time risk management system?**
5. **Design a regulatory compliance system for financial institutions.**

## Conclusion

Fintech is a complex domain that requires deep understanding of:

- **Financial Systems**: Payment processing, banking, accounting
- **Security**: Encryption, fraud detection, compliance
- **Regulations**: KYC/AML, PCI DSS, GDPR
- **Technology**: Blockchain, real-time processing, data analytics
- **Risk Management**: Credit scoring, fraud detection, compliance monitoring

Key skills for fintech backend engineers:
- Strong understanding of financial concepts
- Expertise in security and compliance
- Experience with high-volume, low-latency systems
- Knowledge of regulatory requirements
- Ability to design fault-tolerant systems

This guide provides a foundation for understanding fintech systems and preparing for fintech engineering interviews.
