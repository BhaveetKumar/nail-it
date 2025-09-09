# Strategy Pattern

## Pattern Name & Intent

**Strategy** - Define a family of algorithms, encapsulate each one, and make them interchangeable.

The Strategy pattern defines a family of algorithms, encapsulates each algorithm, and makes them interchangeable. This pattern lets the algorithm vary independently from clients that use it. It's particularly useful when you have multiple ways to perform a task and want to choose the algorithm at runtime.

## When to Use

### Appropriate Scenarios

- **Multiple Algorithms**: When you have multiple ways to perform the same task
- **Runtime Selection**: When algorithm selection depends on runtime conditions
- **Algorithm Variation**: When algorithms need to be easily swapped
- **Avoiding Conditionals**: When you want to eliminate large conditional statements
- **Open/Closed Principle**: When you want to add new algorithms without modifying existing code

### When NOT to Use

- **Simple Algorithms**: When you only have one way to perform a task
- **Performance Critical**: When strategy selection overhead is too high
- **Tight Coupling**: When algorithms are tightly coupled to specific contexts
- **Static Selection**: When algorithm selection is known at compile time

## Real-World Use Cases (Fintech/Payments)

### Payment Processing Strategies

```go
// Payment processing strategy interface
type PaymentStrategy interface {
    ProcessPayment(amount float64, currency string, paymentData map[string]interface{}) (*PaymentResult, error)
    ValidatePayment(paymentData map[string]interface{}) error
    GetStrategyName() string
}

// Payment result structure
type PaymentResult struct {
    TransactionID string  `json:"transaction_id"`
    Status        string  `json:"status"`
    Amount        float64 `json:"amount"`
    Currency      string  `json:"currency"`
    Strategy      string  `json:"strategy"`
    ProcessedAt   time.Time `json:"processed_at"`
}

// Credit card payment strategy
type CreditCardStrategy struct {
    apiKey string
}

func NewCreditCardStrategy(apiKey string) *CreditCardStrategy {
    return &CreditCardStrategy{apiKey: apiKey}
}

func (ccs *CreditCardStrategy) ProcessPayment(amount float64, currency string, paymentData map[string]interface{}) (*PaymentResult, error) {
    // Validate credit card data
    if err := ccs.ValidatePayment(paymentData); err != nil {
        return nil, err
    }

    // Simulate credit card processing
    cardNumber := paymentData["card_number"].(string)
    expiryDate := paymentData["expiry_date"].(string)
    cvv := paymentData["cvv"].(string)

    log.Printf("Processing credit card payment: %s, Amount: %.2f %s",
        maskCardNumber(cardNumber), amount, currency)

    return &PaymentResult{
        TransactionID: "cc_" + generateID(),
        Status:        "success",
        Amount:        amount,
        Currency:      currency,
        Strategy:      "credit_card",
        ProcessedAt:   time.Now(),
    }, nil
}

func (ccs *CreditCardStrategy) ValidatePayment(paymentData map[string]interface{}) error {
    requiredFields := []string{"card_number", "expiry_date", "cvv"}
    for _, field := range requiredFields {
        if _, exists := paymentData[field]; !exists {
            return fmt.Errorf("missing required field: %s", field)
        }
    }

    // Validate card number format
    cardNumber := paymentData["card_number"].(string)
    if !isValidCardNumber(cardNumber) {
        return fmt.Errorf("invalid card number format")
    }

    return nil
}

func (ccs *CreditCardStrategy) GetStrategyName() string {
    return "credit_card"
}

// Bank transfer strategy
type BankTransferStrategy struct {
    bankAPIKey string
}

func NewBankTransferStrategy(bankAPIKey string) *BankTransferStrategy {
    return &BankTransferStrategy{bankAPIKey: bankAPIKey}
}

func (bts *BankTransferStrategy) ProcessPayment(amount float64, currency string, paymentData map[string]interface{}) (*PaymentResult, error) {
    // Validate bank transfer data
    if err := bts.ValidatePayment(paymentData); err != nil {
        return nil, err
    }

    // Simulate bank transfer processing
    accountNumber := paymentData["account_number"].(string)
    routingNumber := paymentData["routing_number"].(string)

    log.Printf("Processing bank transfer: Account: %s, Amount: %.2f %s",
        maskAccountNumber(accountNumber), amount, currency)

    return &PaymentResult{
        TransactionID: "bt_" + generateID(),
        Status:        "success",
        Amount:        amount,
        Currency:      currency,
        Strategy:      "bank_transfer",
        ProcessedAt:   time.Now(),
    }, nil
}

func (bts *BankTransferStrategy) ValidatePayment(paymentData map[string]interface{}) error {
    requiredFields := []string{"account_number", "routing_number"}
    for _, field := range requiredFields {
        if _, exists := paymentData[field]; !exists {
            return fmt.Errorf("missing required field: %s", field)
        }
    }

    return nil
}

func (bts *BankTransferStrategy) GetStrategyName() string {
    return "bank_transfer"
}

// Digital wallet strategy
type DigitalWalletStrategy struct {
    walletProvider string
    apiKey         string
}

func NewDigitalWalletStrategy(walletProvider, apiKey string) *DigitalWalletStrategy {
    return &DigitalWalletStrategy{
        walletProvider: walletProvider,
        apiKey:         apiKey,
    }
}

func (dws *DigitalWalletStrategy) ProcessPayment(amount float64, currency string, paymentData map[string]interface{}) (*PaymentResult, error) {
    // Validate wallet data
    if err := dws.ValidatePayment(paymentData); err != nil {
        return nil, err
    }

    // Simulate digital wallet processing
    walletID := paymentData["wallet_id"].(string)
    pin := paymentData["pin"].(string)

    log.Printf("Processing digital wallet payment: Wallet: %s, Amount: %.2f %s",
        maskWalletID(walletID), amount, currency)

    return &PaymentResult{
        TransactionID: "dw_" + generateID(),
        Status:        "success",
        Amount:        amount,
        Currency:      currency,
        Strategy:      "digital_wallet",
        ProcessedAt:   time.Now(),
    }, nil
}

func (dws *DigitalWalletStrategy) ValidatePayment(paymentData map[string]interface{}) error {
    requiredFields := []string{"wallet_id", "pin"}
    for _, field := range requiredFields {
        if _, exists := paymentData[field]; !exists {
            return fmt.Errorf("missing required field: %s", field)
        }
    }

    return nil
}

func (dws *DigitalWalletStrategy) GetStrategyName() string {
    return "digital_wallet"
}

// Payment processor context
type PaymentProcessor struct {
    strategy PaymentStrategy
}

func NewPaymentProcessor(strategy PaymentStrategy) *PaymentProcessor {
    return &PaymentProcessor{strategy: strategy}
}

func (pp *PaymentProcessor) SetStrategy(strategy PaymentStrategy) {
    pp.strategy = strategy
}

func (pp *PaymentProcessor) ProcessPayment(amount float64, currency string, paymentData map[string]interface{}) (*PaymentResult, error) {
    if pp.strategy == nil {
        return nil, fmt.Errorf("no payment strategy set")
    }

    return pp.strategy.ProcessPayment(amount, currency, paymentData)
}

func (pp *PaymentProcessor) ValidatePayment(paymentData map[string]interface{}) error {
    if pp.strategy == nil {
        return fmt.Errorf("no payment strategy set")
    }

    return pp.strategy.ValidatePayment(paymentData)
}
```

### Risk Assessment Strategies

```go
// Risk assessment strategy interface
type RiskAssessmentStrategy interface {
    AssessRisk(transaction *Transaction) (*RiskAssessment, error)
    GetStrategyName() string
}

// Transaction structure
type Transaction struct {
    ID          string    `json:"id"`
    Amount      float64   `json:"amount"`
    Currency    string    `json:"currency"`
    UserID      string    `json:"user_id"`
    MerchantID  string    `json:"merchant_id"`
    Country     string    `json:"country"`
    IPAddress   string    `json:"ip_address"`
    DeviceID    string    `json:"device_id"`
    Timestamp   time.Time `json:"timestamp"`
}

// Risk assessment result
type RiskAssessment struct {
    RiskScore   int    `json:"risk_score"`
    RiskLevel   string `json:"risk_level"`
    Reasons     []string `json:"reasons"`
    Strategy    string `json:"strategy"`
    AssessedAt  time.Time `json:"assessed_at"`
}

// Basic risk assessment strategy
type BasicRiskStrategy struct{}

func NewBasicRiskStrategy() *BasicRiskStrategy {
    return &BasicRiskStrategy{}
}

func (brs *BasicRiskStrategy) AssessRisk(transaction *Transaction) (*RiskAssessment, error) {
    riskScore := 0
    reasons := []string{}

    // Amount-based risk
    if transaction.Amount > 10000 {
        riskScore += 30
        reasons = append(reasons, "High transaction amount")
    } else if transaction.Amount > 1000 {
        riskScore += 10
        reasons = append(reasons, "Medium transaction amount")
    }

    // Country-based risk
    highRiskCountries := []string{"XX", "YY", "ZZ"}
    for _, country := range highRiskCountries {
        if transaction.Country == country {
            riskScore += 20
            reasons = append(reasons, "High-risk country")
            break
        }
    }

    // Determine risk level
    riskLevel := "low"
    if riskScore >= 50 {
        riskLevel = "high"
    } else if riskScore >= 20 {
        riskLevel = "medium"
    }

    return &RiskAssessment{
        RiskScore:  riskScore,
        RiskLevel:  riskLevel,
        Reasons:    reasons,
        Strategy:   "basic",
        AssessedAt: time.Now(),
    }, nil
}

func (brs *BasicRiskStrategy) GetStrategyName() string {
    return "basic"
}

// Machine learning risk assessment strategy
type MLRiskStrategy struct {
    modelPath string
}

func NewMLRiskStrategy(modelPath string) *MLRiskStrategy {
    return &MLRiskStrategy{modelPath: modelPath}
}

func (mlrs *MLRiskStrategy) AssessRisk(transaction *Transaction) (*RiskAssessment, error) {
    // Simulate ML model prediction
    features := []float64{
        float64(transaction.Amount),
        float64(len(transaction.UserID)),
        float64(len(transaction.MerchantID)),
    }

    // Simulate ML prediction (in real implementation, this would call the ML model)
    riskScore := int(mlrs.predictRisk(features))
    reasons := []string{"ML model prediction"}

    // Determine risk level
    riskLevel := "low"
    if riskScore >= 70 {
        riskLevel = "high"
    } else if riskScore >= 40 {
        riskLevel = "medium"
    }

    return &RiskAssessment{
        RiskScore:  riskScore,
        RiskLevel:  riskLevel,
        Reasons:    reasons,
        Strategy:   "ml",
        AssessedAt: time.Now(),
    }, nil
}

func (mlrs *MLRiskStrategy) predictRisk(features []float64) float64 {
    // Simulate ML prediction
    return 25.0 + float64(len(features))*10.0
}

func (mlrs *MLRiskStrategy) GetStrategyName() string {
    return "ml"
}

// Risk assessor context
type RiskAssessor struct {
    strategy RiskAssessmentStrategy
}

func NewRiskAssessor(strategy RiskAssessmentStrategy) *RiskAssessor {
    return &RiskAssessor{strategy: strategy}
}

func (ra *RiskAssessor) SetStrategy(strategy RiskAssessmentStrategy) {
    ra.strategy = strategy
}

func (ra *RiskAssessor) AssessRisk(transaction *Transaction) (*RiskAssessment, error) {
    if ra.strategy == nil {
        return nil, fmt.Errorf("no risk assessment strategy set")
    }

    return ra.strategy.AssessRisk(transaction)
}
```

## Go Implementation

### Generic Strategy Pattern

```go
package main

import (
    "fmt"
    "sync"
)

// Generic strategy interface
type Strategy[T any, R any] interface {
    Execute(input T) (R, error)
    GetName() string
}

// Generic strategy context
type StrategyContext[T any, R any] struct {
    strategy Strategy[T, R]
    mutex    sync.RWMutex
}

func NewStrategyContext[T any, R any]() *StrategyContext[T, R] {
    return &StrategyContext[T, R]{}
}

func (sc *StrategyContext[T, R]) SetStrategy(strategy Strategy[T, R]) {
    sc.mutex.Lock()
    defer sc.mutex.Unlock()
    sc.strategy = strategy
}

func (sc *StrategyContext[T, R]) Execute(input T) (R, error) {
    sc.mutex.RLock()
    strategy := sc.strategy
    sc.mutex.RUnlock()

    if strategy == nil {
        var zero R
        return zero, fmt.Errorf("no strategy set")
    }

    return strategy.Execute(input)
}

// Sorting strategy example
type SortingStrategy interface {
    Sort(data []int) []int
    GetName() string
}

type BubbleSortStrategy struct{}

func (bss *BubbleSortStrategy) Sort(data []int) []int {
    n := len(data)
    result := make([]int, n)
    copy(result, data)

    for i := 0; i < n-1; i++ {
        for j := 0; j < n-i-1; j++ {
            if result[j] > result[j+1] {
                result[j], result[j+1] = result[j+1], result[j]
            }
        }
    }

    return result
}

func (bss *BubbleSortStrategy) GetName() string {
    return "bubble_sort"
}

type QuickSortStrategy struct{}

func (qss *QuickSortStrategy) Sort(data []int) []int {
    result := make([]int, len(data))
    copy(result, data)
    qss.quickSort(result, 0, len(result)-1)
    return result
}

func (qss *QuickSortStrategy) quickSort(data []int, low, high int) {
    if low < high {
        pi := qss.partition(data, low, high)
        qss.quickSort(data, low, pi-1)
        qss.quickSort(data, pi+1, high)
    }
}

func (qss *QuickSortStrategy) partition(data []int, low, high int) int {
    pivot := data[high]
    i := low - 1

    for j := low; j < high; j++ {
        if data[j] < pivot {
            i++
            data[i], data[j] = data[j], data[i]
        }
    }
    data[i+1], data[high] = data[high], data[i+1]
    return i + 1
}

func (qss *QuickSortStrategy) GetName() string {
    return "quick_sort"
}

type MergeSortStrategy struct{}

func (mss *MergeSortStrategy) Sort(data []int) []int {
    if len(data) <= 1 {
        return data
    }

    mid := len(data) / 2
    left := mss.Sort(data[:mid])
    right := mss.Sort(data[mid:])

    return mss.merge(left, right)
}

func (mss *MergeSortStrategy) merge(left, right []int) []int {
    result := make([]int, 0, len(left)+len(right))
    i, j := 0, 0

    for i < len(left) && j < len(right) {
        if left[i] <= right[j] {
            result = append(result, left[i])
            i++
        } else {
            result = append(result, right[j])
            j++
        }
    }

    result = append(result, left[i:]...)
    result = append(result, right[j:]...)

    return result
}

func (mss *MergeSortStrategy) GetName() string {
    return "merge_sort"
}

// Sorting context
type SortingContext struct {
    strategy SortingStrategy
}

func NewSortingContext() *SortingContext {
    return &SortingContext{}
}

func (sc *SortingContext) SetStrategy(strategy SortingStrategy) {
    sc.strategy = strategy
}

func (sc *SortingContext) Sort(data []int) ([]int, error) {
    if sc.strategy == nil {
        return nil, fmt.Errorf("no sorting strategy set")
    }

    return sc.strategy.Sort(data), nil
}
```

### Strategy with Factory

```go
// Strategy factory
type StrategyFactory[T any, R any] struct {
    strategies map[string]func() Strategy[T, R]
}

func NewStrategyFactory[T any, R any]() *StrategyFactory[T, R] {
    return &StrategyFactory[T, R]{
        strategies: make(map[string]func() Strategy[T, R]),
    }
}

func (sf *StrategyFactory[T, R]) RegisterStrategy(name string, creator func() Strategy[T, R]) {
    sf.strategies[name] = creator
}

func (sf *StrategyFactory[T, R]) CreateStrategy(name string) (Strategy[T, R], error) {
    creator, exists := sf.strategies[name]
    if !exists {
        return nil, fmt.Errorf("strategy %s not found", name)
    }

    return creator(), nil
}

func (sf *StrategyFactory[T, R]) GetAvailableStrategies() []string {
    strategies := make([]string, 0, len(sf.strategies))
    for name := range sf.strategies {
        strategies = append(strategies, name)
    }
    return strategies
}

// Payment strategy factory
type PaymentStrategyFactory struct {
    *StrategyFactory[PaymentRequest, *PaymentResult]
}

type PaymentRequest struct {
    Amount      float64                `json:"amount"`
    Currency    string                 `json:"currency"`
    PaymentData map[string]interface{} `json:"payment_data"`
}

func NewPaymentStrategyFactory() *PaymentStrategyFactory {
    factory := &PaymentStrategyFactory{
        StrategyFactory: NewStrategyFactory[PaymentRequest, *PaymentResult](),
    }

    // Register default strategies
    factory.RegisterStrategy("credit_card", func() Strategy[PaymentRequest, *PaymentResult] {
        return NewCreditCardStrategy("default_key")
    })

    factory.RegisterStrategy("bank_transfer", func() Strategy[PaymentRequest, *PaymentResult] {
        return NewBankTransferStrategy("default_bank_key")
    })

    factory.RegisterStrategy("digital_wallet", func() Strategy[PaymentRequest, *PaymentResult] {
        return NewDigitalWalletStrategy("default_wallet", "default_key")
    })

    return factory
}
```

## Variants & Trade-offs

### Variants

#### 1. Simple Strategy

```go
type SimpleStrategy interface {
    Execute() error
}

type StrategyA struct{}
func (sa *StrategyA) Execute() error { return nil }

type StrategyB struct{}
func (sb *StrategyB) Execute() error { return nil }
```

**Pros**: Simple and straightforward
**Cons**: Limited flexibility, no input/output parameters

#### 2. Parameterized Strategy

```go
type ParameterizedStrategy interface {
    Execute(params map[string]interface{}) (interface{}, error)
}
```

**Pros**: Flexible input/output
**Cons**: Type safety issues, runtime errors possible

#### 3. Generic Strategy

```go
type GenericStrategy[T any, R any] interface {
    Execute(input T) (R, error)
}
```

**Pros**: Type safety, flexible
**Cons**: More complex, Go generics limitations

### Trade-offs

| Aspect              | Pros                               | Cons                        |
| ------------------- | ---------------------------------- | --------------------------- |
| **Flexibility**     | Easy to add new algorithms         | Can become complex          |
| **Maintainability** | Each strategy is independent       | More classes to maintain    |
| **Testing**         | Easy to test individual strategies | More test cases needed      |
| **Performance**     | Can optimize individual strategies | Strategy selection overhead |
| **Code Reuse**      | Strategies can be reused           | Potential code duplication  |

## Testable Example

```go
package main

import (
    "testing"
)

// Mock payment strategy for testing
type MockPaymentStrategy struct {
    shouldFail bool
    calls      []PaymentRequest
}

func NewMockPaymentStrategy(shouldFail bool) *MockPaymentStrategy {
    return &MockPaymentStrategy{
        shouldFail: shouldFail,
        calls:      make([]PaymentRequest, 0),
    }
}

func (mps *MockPaymentStrategy) ProcessPayment(amount float64, currency string, paymentData map[string]interface{}) (*PaymentResult, error) {
    mps.calls = append(mps.calls, PaymentRequest{
        Amount:      amount,
        Currency:    currency,
        PaymentData: paymentData,
    })

    if mps.shouldFail {
        return nil, fmt.Errorf("payment failed")
    }

    return &PaymentResult{
        TransactionID: "mock_" + generateID(),
        Status:        "success",
        Amount:        amount,
        Currency:      currency,
        Strategy:      "mock",
        ProcessedAt:   time.Now(),
    }, nil
}

func (mps *MockPaymentStrategy) ValidatePayment(paymentData map[string]interface{}) error {
    if mps.shouldFail {
        return fmt.Errorf("validation failed")
    }
    return nil
}

func (mps *MockPaymentStrategy) GetStrategyName() string {
    return "mock"
}

func (mps *MockPaymentStrategy) GetCalls() []PaymentRequest {
    return mps.calls
}

// Tests
func TestPaymentProcessor_CreditCard(t *testing.T) {
    strategy := NewCreditCardStrategy("test_key")
    processor := NewPaymentProcessor(strategy)

    paymentData := map[string]interface{}{
        "card_number": "4111111111111111",
        "expiry_date": "12/25",
        "cvv":         "123",
    }

    result, err := processor.ProcessPayment(100.50, "USD", paymentData)
    if err != nil {
        t.Fatalf("ProcessPayment() error = %v", err)
    }

    if result.Strategy != "credit_card" {
        t.Errorf("Expected strategy 'credit_card', got %s", result.Strategy)
    }

    if result.Amount != 100.50 {
        t.Errorf("Expected amount 100.50, got %.2f", result.Amount)
    }
}

func TestPaymentProcessor_BankTransfer(t *testing.T) {
    strategy := NewBankTransferStrategy("test_bank_key")
    processor := NewPaymentProcessor(strategy)

    paymentData := map[string]interface{}{
        "account_number": "1234567890",
        "routing_number": "987654321",
    }

    result, err := processor.ProcessPayment(500.75, "EUR", paymentData)
    if err != nil {
        t.Fatalf("ProcessPayment() error = %v", err)
    }

    if result.Strategy != "bank_transfer" {
        t.Errorf("Expected strategy 'bank_transfer', got %s", result.Strategy)
    }
}

func TestPaymentProcessor_StrategySwitching(t *testing.T) {
    processor := NewPaymentProcessor(nil)

    // Test with credit card strategy
    ccStrategy := NewCreditCardStrategy("test_key")
    processor.SetStrategy(ccStrategy)

    paymentData := map[string]interface{}{
        "card_number": "4111111111111111",
        "expiry_date": "12/25",
        "cvv":         "123",
    }

    result, err := processor.ProcessPayment(100.50, "USD", paymentData)
    if err != nil {
        t.Fatalf("ProcessPayment() error = %v", err)
    }

    if result.Strategy != "credit_card" {
        t.Errorf("Expected strategy 'credit_card', got %s", result.Strategy)
    }

    // Switch to bank transfer strategy
    btStrategy := NewBankTransferStrategy("test_bank_key")
    processor.SetStrategy(btStrategy)

    paymentData = map[string]interface{}{
        "account_number": "1234567890",
        "routing_number": "987654321",
    }

    result, err = processor.ProcessPayment(200.25, "GBP", paymentData)
    if err != nil {
        t.Fatalf("ProcessPayment() error = %v", err)
    }

    if result.Strategy != "bank_transfer" {
        t.Errorf("Expected strategy 'bank_transfer', got %s", result.Strategy)
    }
}

func TestPaymentProcessor_Validation(t *testing.T) {
    strategy := NewCreditCardStrategy("test_key")
    processor := NewPaymentProcessor(strategy)

    // Test with missing required field
    paymentData := map[string]interface{}{
        "card_number": "4111111111111111",
        "expiry_date": "12/25",
        // Missing CVV
    }

    err := processor.ValidatePayment(paymentData)
    if err == nil {
        t.Error("Expected validation error for missing CVV")
    }
}

func TestSortingContext(t *testing.T) {
    context := NewSortingContext()

    data := []int{64, 34, 25, 12, 22, 11, 90}

    // Test bubble sort
    bubbleSort := &BubbleSortStrategy{}
    context.SetStrategy(bubbleSort)

    result, err := context.Sort(data)
    if err != nil {
        t.Fatalf("Sort() error = %v", err)
    }

    // Verify sorting
    for i := 1; i < len(result); i++ {
        if result[i] < result[i-1] {
            t.Errorf("Array not sorted: %v", result)
        }
    }

    // Test quick sort
    quickSort := &QuickSortStrategy{}
    context.SetStrategy(quickSort)

    result, err = context.Sort(data)
    if err != nil {
        t.Fatalf("Sort() error = %v", err)
    }

    // Verify sorting
    for i := 1; i < len(result); i++ {
        if result[i] < result[i-1] {
            t.Errorf("Array not sorted: %v", result)
        }
    }
}

func TestStrategyFactory(t *testing.T) {
    factory := NewPaymentStrategyFactory()

    // Test creating credit card strategy
    strategy, err := factory.CreateStrategy("credit_card")
    if err != nil {
        t.Fatalf("CreateStrategy() error = %v", err)
    }

    if strategy.GetStrategyName() != "credit_card" {
        t.Errorf("Expected strategy name 'credit_card', got %s", strategy.GetStrategyName())
    }

    // Test creating non-existent strategy
    _, err = factory.CreateStrategy("non_existent")
    if err == nil {
        t.Error("Expected error for non-existent strategy")
    }

    // Test getting available strategies
    strategies := factory.GetAvailableStrategies()
    if len(strategies) == 0 {
        t.Error("Expected available strategies")
    }
}
```

## Integration Tips

### 1. With Configuration

```go
type StrategyConfig struct {
    DefaultStrategy string            `yaml:"default_strategy"`
    Strategies      map[string]string `yaml:"strategies"`
}

func LoadStrategyFromConfig(configPath string) (*PaymentProcessor, error) {
    var config StrategyConfig
    data, err := os.ReadFile(configPath)
    if err != nil {
        return nil, err
    }

    if err := yaml.Unmarshal(data, &config); err != nil {
        return nil, err
    }

    factory := NewPaymentStrategyFactory()
    strategy, err := factory.CreateStrategy(config.DefaultStrategy)
    if err != nil {
        return nil, err
    }

    return NewPaymentProcessor(strategy), nil
}
```

### 2. With Dependency Injection

```go
type PaymentService struct {
    processor *PaymentProcessor
}

func NewPaymentService(processor *PaymentProcessor) *PaymentService {
    return &PaymentService{processor: processor}
}

func (ps *PaymentService) ProcessPayment(paymentType string, amount float64, currency string, paymentData map[string]interface{}) error {
    factory := NewPaymentStrategyFactory()
    strategy, err := factory.CreateStrategy(paymentType)
    if err != nil {
        return err
    }

    ps.processor.SetStrategy(strategy)
    _, err = ps.processor.ProcessPayment(amount, currency, paymentData)
    return err
}
```

### 3. With Middleware

```go
type StrategyMiddleware struct {
    next Strategy[PaymentRequest, *PaymentResult]
}

func (sm *StrategyMiddleware) ProcessPayment(amount float64, currency string, paymentData map[string]interface{}) (*PaymentResult, error) {
    // Pre-processing
    log.Printf("Processing payment: %.2f %s", amount, currency)

    // Call next strategy
    result, err := sm.next.ProcessPayment(amount, currency, paymentData)

    // Post-processing
    if err != nil {
        log.Printf("Payment failed: %v", err)
    } else {
        log.Printf("Payment successful: %s", result.TransactionID)
    }

    return result, err
}

func (sm *StrategyMiddleware) ValidatePayment(paymentData map[string]interface{}) error {
    return sm.next.ValidatePayment(paymentData)
}

func (sm *StrategyMiddleware) GetStrategyName() string {
    return "middleware_" + sm.next.GetStrategyName()
}
```

## Common Interview Questions

### 1. What is the Strategy pattern and when would you use it?

**Answer**: The Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. Use it when you have multiple ways to perform the same task, when algorithm selection depends on runtime conditions, or when you want to eliminate large conditional statements.

### 2. What's the difference between Strategy and State patterns?

**Answer**: Strategy pattern focuses on different algorithms for the same task, while State pattern focuses on different behaviors based on object state. Strategy is about "how" to do something, while State is about "what" to do based on current state.

### 3. How do you implement the Strategy pattern in Go?

**Answer**: Define a strategy interface with the common method signature, create concrete strategy implementations, implement a context class that uses the strategy, and provide a way to set/change the strategy at runtime.

### 4. What are the benefits and drawbacks of the Strategy pattern?

**Answer**: Benefits include flexibility to add new algorithms, elimination of conditional statements, and easy testing of individual strategies. Drawbacks include increased number of classes, potential code duplication, and strategy selection overhead.

### 5. How do you choose between Strategy and Factory patterns?

**Answer**: Use Strategy when you have multiple algorithms for the same task and want to choose at runtime. Use Factory when you need to create objects of different types. Strategy is about behavior, Factory is about object creation.
