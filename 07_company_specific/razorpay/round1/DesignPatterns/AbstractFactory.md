---
# Auto-generated front matter
Title: Abstractfactory
LastUpdated: 2025-11-06T20:45:58.514332
Tags: []
Status: draft
---

# Abstract Factory Pattern

## Pattern Name & Intent

**Abstract Factory** is a creational design pattern that provides an interface for creating families of related or dependent objects without specifying their concrete classes. It encapsulates a group of individual factories that have a common theme.

**Key Intent:**

- Create families of related products
- Ensure products from a family are used together
- Provide abstraction over concrete product creation
- Support multiple product families through different factory implementations

## When to Use

**Use Abstract Factory when:**

1. **Multiple Product Families**: Your system needs to work with multiple families of related products
2. **Product Consistency**: You want to ensure that products from one family are used together
3. **Runtime Selection**: You need to select product families at runtime based on configuration
4. **Platform Independence**: You want to abstract platform-specific implementations
5. **Complex Object Creation**: You have complex object creation logic that varies by family

**Don't use when:**

- You only have one product family (use Factory Method instead)
- Product families rarely change (adds unnecessary complexity)
- Simple object creation is sufficient

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Gateway Abstraction

```go
// Different payment gateway families with consistent interfaces
type PaymentGatewayFactory interface {
    CreateProcessor() PaymentProcessor
    CreateValidator() PaymentValidator
    CreateLogger() PaymentLogger
}

// Razorpay family
type RazorpayFactory struct{}
func (r *RazorpayFactory) CreateProcessor() PaymentProcessor { return &RazorpayProcessor{} }
func (r *RazorpayFactory) CreateValidator() PaymentValidator { return &RazorpayValidator{} }
func (r *RazorpayFactory) CreateLogger() PaymentLogger { return &RazorpayLogger{} }

// Stripe family
type StripeFactory struct{}
func (s *StripeFactory) CreateProcessor() PaymentProcessor { return &StripeProcessor{} }
func (s *StripeFactory) CreateValidator() PaymentValidator { return &StripeValidator{} }
func (s *StripeFactory) CreateLogger() PaymentLogger { return &StripeLogger{} }
```

### 2. Multi-Region Banking Systems

```go
// Different regulatory compliance families
type BankingSystemFactory interface {
    CreateAccountManager() AccountManager
    CreateComplianceChecker() ComplianceChecker
    CreateReportGenerator() ReportGenerator
}

// US Banking (SOX, FDIC compliance)
type USBankingFactory struct{}

// EU Banking (GDPR, PSD2 compliance)
type EUBankingFactory struct{}

// India Banking (RBI compliance)
type IndiaBankingFactory struct{}
```

### 3. Multi-Currency Trading Platforms

```go
// Different market families with specific rules
type TradingPlatformFactory interface {
    CreateOrderProcessor() OrderProcessor
    CreateRiskManager() RiskManager
    CreateMarketDataProvider() MarketDataProvider
}

// Equity trading family
type EquityTradingFactory struct{}

// Cryptocurrency trading family
type CryptoTradingFactory struct{}

// Forex trading family
type ForexTradingFactory struct{}
```

### 4. Multi-Tenant SaaS Platforms

```go
// Different tenant tiers with varying capabilities
type TenantServiceFactory interface {
    CreateUserManager() UserManager
    CreateStorageManager() StorageManager
    CreateAnalyticsProvider() AnalyticsProvider
}

// Enterprise tier
type EnterpriseTenantFactory struct{}

// Professional tier
type ProfessionalTenantFactory struct{}

// Basic tier
type BasicTenantFactory struct{}
```

## Go Implementation

```go
package main

import (
    "fmt"
    "log"
    "time"
)

// Product interfaces
type PaymentProcessor interface {
    ProcessPayment(amount float64, currency string) error
    GetTransactionFee(amount float64) float64
    SupportedCurrencies() []string
}

type PaymentValidator interface {
    ValidateCard(cardNumber string) error
    ValidateAmount(amount float64, currency string) error
    ValidateRiskProfile(userID string, amount float64) error
}

type PaymentLogger interface {
    LogTransaction(transactionID string, amount float64, status string)
    LogError(transactionID string, err error)
    LogMetrics(processingTime time.Duration, gateway string)
}

// Abstract Factory interface
type PaymentGatewayFactory interface {
    CreateProcessor() PaymentProcessor
    CreateValidator() PaymentValidator
    CreateLogger() PaymentLogger
    GetGatewayName() string
    GetSupportedRegions() []string
}

// Razorpay product family
type RazorpayProcessor struct {
    apiKey    string
    apiSecret string
}

func (r *RazorpayProcessor) ProcessPayment(amount float64, currency string) error {
    // Razorpay-specific payment processing
    fmt.Printf("Processing ₹%.2f payment through Razorpay\n", amount)

    if amount <= 0 {
        return fmt.Errorf("invalid amount: %.2f", amount)
    }

    // Simulate API call delay
    time.Sleep(100 * time.Millisecond)

    return nil
}

func (r *RazorpayProcessor) GetTransactionFee(amount float64) float64 {
    // Razorpay fee structure: 2% + ₹2
    return (amount * 0.02) + 2.0
}

func (r *RazorpayProcessor) SupportedCurrencies() []string {
    return []string{"INR", "USD", "EUR"}
}

type RazorpayValidator struct {
    riskThreshold float64
}

func (r *RazorpayValidator) ValidateCard(cardNumber string) error {
    if len(cardNumber) < 13 || len(cardNumber) > 19 {
        return fmt.Errorf("invalid card number length")
    }

    // Luhn algorithm validation would go here
    fmt.Printf("Validating card through Razorpay: %s***\n", cardNumber[:4])
    return nil
}

func (r *RazorpayValidator) ValidateAmount(amount float64, currency string) error {
    if currency == "INR" && amount > 200000 {
        return fmt.Errorf("amount exceeds daily limit for INR: %.2f", amount)
    }
    return nil
}

func (r *RazorpayValidator) ValidateRiskProfile(userID string, amount float64) error {
    // Razorpay-specific risk assessment
    if amount > r.riskThreshold {
        return fmt.Errorf("transaction flagged for manual review: user %s, amount %.2f", userID, amount)
    }
    return nil
}

type RazorpayLogger struct {
    logFormat string
}

func (r *RazorpayLogger) LogTransaction(transactionID string, amount float64, status string) {
    log.Printf("[RAZORPAY] Transaction %s: ₹%.2f - %s", transactionID, amount, status)
}

func (r *RazorpayLogger) LogError(transactionID string, err error) {
    log.Printf("[RAZORPAY ERROR] Transaction %s: %v", transactionID, err)
}

func (r *RazorpayLogger) LogMetrics(processingTime time.Duration, gateway string) {
    log.Printf("[RAZORPAY METRICS] Gateway: %s, Processing Time: %v", gateway, processingTime)
}

// Razorpay Factory
type RazorpayFactory struct {
    config RazorpayConfig
}

type RazorpayConfig struct {
    APIKey        string
    APISecret     string
    RiskThreshold float64
    LogFormat     string
}

func NewRazorpayFactory(config RazorpayConfig) *RazorpayFactory {
    return &RazorpayFactory{config: config}
}

func (r *RazorpayFactory) CreateProcessor() PaymentProcessor {
    return &RazorpayProcessor{
        apiKey:    r.config.APIKey,
        apiSecret: r.config.APISecret,
    }
}

func (r *RazorpayFactory) CreateValidator() PaymentValidator {
    return &RazorpayValidator{
        riskThreshold: r.config.RiskThreshold,
    }
}

func (r *RazorpayFactory) CreateLogger() PaymentLogger {
    return &RazorpayLogger{
        logFormat: r.config.LogFormat,
    }
}

func (r *RazorpayFactory) GetGatewayName() string {
    return "Razorpay"
}

func (r *RazorpayFactory) GetSupportedRegions() []string {
    return []string{"India", "Southeast Asia"}
}

// Stripe product family
type StripeProcessor struct {
    secretKey string
    publicKey string
}

func (s *StripeProcessor) ProcessPayment(amount float64, currency string) error {
    fmt.Printf("Processing $%.2f payment through Stripe\n", amount)

    if amount <= 0 {
        return fmt.Errorf("invalid amount: %.2f", amount)
    }

    // Simulate API call delay
    time.Sleep(150 * time.Millisecond)

    return nil
}

func (s *StripeProcessor) GetTransactionFee(amount float64) float64 {
    // Stripe fee structure: 2.9% + $0.30
    return (amount * 0.029) + 0.30
}

func (s *StripeProcessor) SupportedCurrencies() []string {
    return []string{"USD", "EUR", "GBP", "CAD", "AUD"}
}

type StripeValidator struct {
    fraudDetection bool
}

func (s *StripeValidator) ValidateCard(cardNumber string) error {
    if len(cardNumber) < 13 || len(cardNumber) > 19 {
        return fmt.Errorf("invalid card number length")
    }

    fmt.Printf("Validating card through Stripe: %s***\n", cardNumber[:4])
    return nil
}

func (s *StripeValidator) ValidateAmount(amount float64, currency string) error {
    if currency == "USD" && amount > 999999.99 {
        return fmt.Errorf("amount exceeds maximum limit for USD: %.2f", amount)
    }
    return nil
}

func (s *StripeValidator) ValidateRiskProfile(userID string, amount float64) error {
    if s.fraudDetection && amount > 10000 {
        return fmt.Errorf("high-value transaction requires additional verification: user %s", userID)
    }
    return nil
}

type StripeLogger struct {
    webhookURL string
}

func (s *StripeLogger) LogTransaction(transactionID string, amount float64, status string) {
    log.Printf("[STRIPE] Transaction %s: $%.2f - %s", transactionID, amount, status)
}

func (s *StripeLogger) LogError(transactionID string, err error) {
    log.Printf("[STRIPE ERROR] Transaction %s: %v", transactionID, err)
}

func (s *StripeLogger) LogMetrics(processingTime time.Duration, gateway string) {
    log.Printf("[STRIPE METRICS] Gateway: %s, Processing Time: %v", gateway, processingTime)
}

// Stripe Factory
type StripeFactory struct {
    config StripeConfig
}

type StripeConfig struct {
    SecretKey       string
    PublicKey       string
    WebhookURL      string
    FraudDetection  bool
}

func NewStripeFactory(config StripeConfig) *StripeFactory {
    return &StripeFactory{config: config}
}

func (s *StripeFactory) CreateProcessor() PaymentProcessor {
    return &StripeProcessor{
        secretKey: s.config.SecretKey,
        publicKey: s.config.PublicKey,
    }
}

func (s *StripeFactory) CreateValidator() PaymentValidator {
    return &StripeValidator{
        fraudDetection: s.config.FraudDetection,
    }
}

func (s *StripeFactory) CreateLogger() PaymentLogger {
    return &StripeLogger{
        webhookURL: s.config.WebhookURL,
    }
}

func (s *StripeFactory) GetGatewayName() string {
    return "Stripe"
}

func (s *StripeFactory) GetSupportedRegions() []string {
    return []string{"Global", "North America", "Europe"}
}

// Payment Service using Abstract Factory
type PaymentService struct {
    factory   PaymentGatewayFactory
    processor PaymentProcessor
    validator PaymentValidator
    logger    PaymentLogger
}

func NewPaymentService(factory PaymentGatewayFactory) *PaymentService {
    return &PaymentService{
        factory:   factory,
        processor: factory.CreateProcessor(),
        validator: factory.CreateValidator(),
        logger:    factory.CreateLogger(),
    }
}

func (p *PaymentService) ProcessTransaction(transactionID, userID, cardNumber string, amount float64, currency string) error {
    start := time.Now()

    // Validate card
    if err := p.validator.ValidateCard(cardNumber); err != nil {
        p.logger.LogError(transactionID, err)
        return fmt.Errorf("card validation failed: %w", err)
    }

    // Validate amount
    if err := p.validator.ValidateAmount(amount, currency); err != nil {
        p.logger.LogError(transactionID, err)
        return fmt.Errorf("amount validation failed: %w", err)
    }

    // Validate risk profile
    if err := p.validator.ValidateRiskProfile(userID, amount); err != nil {
        p.logger.LogError(transactionID, err)
        return fmt.Errorf("risk validation failed: %w", err)
    }

    // Process payment
    if err := p.processor.ProcessPayment(amount, currency); err != nil {
        p.logger.LogError(transactionID, err)
        return fmt.Errorf("payment processing failed: %w", err)
    }

    // Log successful transaction
    p.logger.LogTransaction(transactionID, amount, "SUCCESS")
    p.logger.LogMetrics(time.Since(start), p.factory.GetGatewayName())

    return nil
}

func (p *PaymentService) GetTransactionFee(amount float64) float64 {
    return p.processor.GetTransactionFee(amount)
}

func (p *PaymentService) GetSupportedCurrencies() []string {
    return p.processor.SupportedCurrencies()
}

// Factory Manager for runtime factory selection
type FactoryManager struct {
    factories map[string]PaymentGatewayFactory
}

func NewFactoryManager() *FactoryManager {
    return &FactoryManager{
        factories: make(map[string]PaymentGatewayFactory),
    }
}

func (f *FactoryManager) RegisterFactory(name string, factory PaymentGatewayFactory) {
    f.factories[name] = factory
}

func (f *FactoryManager) GetFactory(name string) (PaymentGatewayFactory, error) {
    factory, exists := f.factories[name]
    if !exists {
        return nil, fmt.Errorf("factory not found: %s", name)
    }
    return factory, nil
}

func (f *FactoryManager) GetAvailableFactories() []string {
    var names []string
    for name := range f.factories {
        names = append(names, name)
    }
    return names
}

// Example usage
func main() {
    // Configure factories
    razorpayConfig := RazorpayConfig{
        APIKey:        "rzp_test_key",
        APISecret:     "rzp_test_secret",
        RiskThreshold: 50000,
        LogFormat:     "json",
    }

    stripeConfig := StripeConfig{
        SecretKey:      "sk_test_key",
        PublicKey:      "pk_test_key",
        WebhookURL:     "https://api.mysite.com/stripe/webhook",
        FraudDetection: true,
    }

    // Create factories
    razorpayFactory := NewRazorpayFactory(razorpayConfig)
    stripeFactory := NewStripeFactory(stripeConfig)

    // Register factories
    factoryManager := NewFactoryManager()
    factoryManager.RegisterFactory("razorpay", razorpayFactory)
    factoryManager.RegisterFactory("stripe", stripeFactory)

    // Example 1: Use Razorpay for Indian market
    fmt.Println("=== Processing Indian Payment ===")
    razorpayService := NewPaymentService(razorpayFactory)

    err := razorpayService.ProcessTransaction(
        "TXN_001",
        "USER_123",
        "4111111111111111",
        15000,
        "INR",
    )

    if err != nil {
        fmt.Printf("Transaction failed: %v\n", err)
    } else {
        fmt.Printf("Transaction successful! Fee: ₹%.2f\n",
            razorpayService.GetTransactionFee(15000))
    }

    fmt.Println()

    // Example 2: Use Stripe for US market
    fmt.Println("=== Processing US Payment ===")
    stripeService := NewPaymentService(stripeFactory)

    err = stripeService.ProcessTransaction(
        "TXN_002",
        "USER_456",
        "4242424242424242",
        299.99,
        "USD",
    )

    if err != nil {
        fmt.Printf("Transaction failed: %v\n", err)
    } else {
        fmt.Printf("Transaction successful! Fee: $%.2f\n",
            stripeService.GetTransactionFee(299.99))
    }

    fmt.Println()

    // Example 3: Runtime factory selection
    fmt.Println("=== Runtime Factory Selection ===")

    region := "US"
    var factoryName string

    switch region {
    case "India":
        factoryName = "razorpay"
    case "US", "Europe":
        factoryName = "stripe"
    default:
        factoryName = "stripe" // default
    }

    factory, err := factoryManager.GetFactory(factoryName)
    if err != nil {
        log.Fatal(err)
    }

    service := NewPaymentService(factory)
    fmt.Printf("Selected gateway: %s for region: %s\n",
        factory.GetGatewayName(), region)
    fmt.Printf("Supported currencies: %v\n",
        service.GetSupportedCurrencies())
    fmt.Printf("Supported regions: %v\n",
        factory.GetSupportedRegions())
}
```

## Variants & Trade-offs

### Variants

1. **Registry-Based Abstract Factory**

```go
type FactoryRegistry struct {
    factories map[string]PaymentGatewayFactory
}

func (r *FactoryRegistry) Register(name string, factory PaymentGatewayFactory) {
    r.factories[name] = factory
}

func (r *FactoryRegistry) Create(name string) PaymentGatewayFactory {
    return r.factories[name]
}
```

2. **Configuration-Driven Factory**

```go
type ConfigurableFactory struct {
    config FactoryConfig
}

func (c *ConfigurableFactory) CreateFactory() PaymentGatewayFactory {
    switch c.config.Type {
    case "razorpay":
        return NewRazorpayFactory(c.config.RazorpayConfig)
    case "stripe":
        return NewStripeFactory(c.config.StripeConfig)
    }
    return nil
}
```

3. **Plugin-Based Factory**

```go
type PluginFactory interface {
    PaymentGatewayFactory
    LoadPlugin(path string) error
    UnloadPlugin() error
}
```

### Trade-offs

**Pros:**

- **Family Consistency**: Ensures related products work together
- **Easy Switching**: Can switch entire product families easily
- **Runtime Selection**: Supports runtime factory selection
- **Extensibility**: Easy to add new product families
- **Isolation**: Each family's implementation is isolated

**Cons:**

- **Complexity**: More complex than simple Factory Method
- **Interface Proliferation**: Many interfaces to maintain
- **Overhead**: Additional abstraction layers
- **Learning Curve**: More complex to understand and implement
- **Over-engineering**: Can be overkill for simple scenarios

**When to Choose Abstract Factory vs Alternatives:**

| Pattern          | Use When                          | Avoid When                |
| ---------------- | --------------------------------- | ------------------------- |
| Abstract Factory | Multiple related product families | Single product family     |
| Factory Method   | Single product with variants      | Multiple related products |
| Builder          | Complex object construction       | Simple object creation    |
| Prototype        | Expensive object creation         | Cheap object creation     |

## Testable Example

```go
package main

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
)

// Mock implementations for testing
type MockProcessor struct {
    mock.Mock
}

func (m *MockProcessor) ProcessPayment(amount float64, currency string) error {
    args := m.Called(amount, currency)
    return args.Error(0)
}

func (m *MockProcessor) GetTransactionFee(amount float64) float64 {
    args := m.Called(amount)
    return args.Get(0).(float64)
}

func (m *MockProcessor) SupportedCurrencies() []string {
    args := m.Called()
    return args.Get(0).([]string)
}

type MockValidator struct {
    mock.Mock
}

func (m *MockValidator) ValidateCard(cardNumber string) error {
    args := m.Called(cardNumber)
    return args.Error(0)
}

func (m *MockValidator) ValidateAmount(amount float64, currency string) error {
    args := m.Called(amount, currency)
    return args.Error(0)
}

func (m *MockValidator) ValidateRiskProfile(userID string, amount float64) error {
    args := m.Called(userID, amount)
    return args.Error(0)
}

type MockLogger struct {
    mock.Mock
}

func (m *MockLogger) LogTransaction(transactionID string, amount float64, status string) {
    m.Called(transactionID, amount, status)
}

func (m *MockLogger) LogError(transactionID string, err error) {
    m.Called(transactionID, err)
}

func (m *MockLogger) LogMetrics(processingTime time.Duration, gateway string) {
    m.Called(processingTime, gateway)
}

type MockFactory struct {
    mock.Mock
    processor PaymentProcessor
    validator PaymentValidator
    logger    PaymentLogger
}

func (m *MockFactory) CreateProcessor() PaymentProcessor {
    return m.processor
}

func (m *MockFactory) CreateValidator() PaymentValidator {
    return m.validator
}

func (m *MockFactory) CreateLogger() PaymentLogger {
    return m.logger
}

func (m *MockFactory) GetGatewayName() string {
    args := m.Called()
    return args.String(0)
}

func (m *MockFactory) GetSupportedRegions() []string {
    args := m.Called()
    return args.Get(0).([]string)
}

func TestPaymentService_ProcessTransaction(t *testing.T) {
    // Setup mocks
    mockProcessor := &MockProcessor{}
    mockValidator := &MockValidator{}
    mockLogger := &MockLogger{}
    mockFactory := &MockFactory{
        processor: mockProcessor,
        validator: mockValidator,
        logger:    mockLogger,
    }

    // Setup expectations
    mockValidator.On("ValidateCard", "4111111111111111").Return(nil)
    mockValidator.On("ValidateAmount", 100.0, "USD").Return(nil)
    mockValidator.On("ValidateRiskProfile", "USER_123", 100.0).Return(nil)
    mockProcessor.On("ProcessPayment", 100.0, "USD").Return(nil)
    mockLogger.On("LogTransaction", "TXN_001", 100.0, "SUCCESS").Return()
    mockLogger.On("LogMetrics", mock.AnythingOfType("time.Duration"), "TestGateway").Return()
    mockFactory.On("GetGatewayName").Return("TestGateway")

    // Create service
    service := NewPaymentService(mockFactory)

    // Test successful transaction
    err := service.ProcessTransaction(
        "TXN_001",
        "USER_123",
        "4111111111111111",
        100.0,
        "USD",
    )

    // Assertions
    assert.NoError(t, err)
    mockValidator.AssertExpectations(t)
    mockProcessor.AssertExpectations(t)
    mockLogger.AssertExpectations(t)
    mockFactory.AssertExpectations(t)
}

func TestPaymentService_ProcessTransaction_ValidationFailure(t *testing.T) {
    // Setup mocks
    mockValidator := &MockValidator{}
    mockLogger := &MockLogger{}
    mockFactory := &MockFactory{
        validator: mockValidator,
        logger:    mockLogger,
    }

    // Setup expectations for validation failure
    validationError := fmt.Errorf("invalid card number")
    mockValidator.On("ValidateCard", "invalid").Return(validationError)
    mockLogger.On("LogError", "TXN_002", validationError).Return()

    // Create service
    service := NewPaymentService(mockFactory)

    // Test validation failure
    err := service.ProcessTransaction(
        "TXN_002",
        "USER_123",
        "invalid",
        100.0,
        "USD",
    )

    // Assertions
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "card validation failed")
    mockValidator.AssertExpectations(t)
    mockLogger.AssertExpectations(t)
}

func TestFactoryManager(t *testing.T) {
    manager := NewFactoryManager()

    // Test factory registration
    razorpayFactory := NewRazorpayFactory(RazorpayConfig{})
    manager.RegisterFactory("razorpay", razorpayFactory)

    // Test factory retrieval
    factory, err := manager.GetFactory("razorpay")
    assert.NoError(t, err)
    assert.NotNil(t, factory)
    assert.Equal(t, "Razorpay", factory.GetGatewayName())

    // Test non-existent factory
    _, err = manager.GetFactory("nonexistent")
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "factory not found")

    // Test available factories
    available := manager.GetAvailableFactories()
    assert.Contains(t, available, "razorpay")
}

func TestAbstractFactory_ProductFamilyConsistency(t *testing.T) {
    // Test that Razorpay factory creates Razorpay products
    razorpayFactory := NewRazorpayFactory(RazorpayConfig{
        RiskThreshold: 10000,
    })

    processor := razorpayFactory.CreateProcessor()
    validator := razorpayFactory.CreateValidator()
    logger := razorpayFactory.CreateLogger()

    // Verify product types
    assert.IsType(t, &RazorpayProcessor{}, processor)
    assert.IsType(t, &RazorpayValidator{}, validator)
    assert.IsType(t, &RazorpayLogger{}, logger)

    // Test that Stripe factory creates Stripe products
    stripeFactory := NewStripeFactory(StripeConfig{
        FraudDetection: true,
    })

    processor = stripeFactory.CreateProcessor()
    validator = stripeFactory.CreateValidator()
    logger = stripeFactory.CreateLogger()

    // Verify product types
    assert.IsType(t, &StripeProcessor{}, processor)
    assert.IsType(t, &StripeValidator{}, validator)
    assert.IsType(t, &StripeLogger{}, logger)
}

func BenchmarkPaymentService_ProcessTransaction(b *testing.B) {
    razorpayFactory := NewRazorpayFactory(RazorpayConfig{
        APIKey:        "test_key",
        APISecret:     "test_secret",
        RiskThreshold: 50000,
    })

    service := NewPaymentService(razorpayFactory)

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        err := service.ProcessTransaction(
            fmt.Sprintf("TXN_%d", i),
            "USER_123",
            "4111111111111111",
            100.0,
            "USD",
        )
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

## Integration Tips

### 1. Configuration Management

```go
type FactoryConfig struct {
    Default string                 `yaml:"default"`
    Factories map[string]FactorySpec `yaml:"factories"`
}

type FactorySpec struct {
    Type   string                 `yaml:"type"`
    Config map[string]interface{} `yaml:"config"`
}

func LoadFactoryFromConfig(config FactoryConfig) PaymentGatewayFactory {
    spec := config.Factories[config.Default]

    switch spec.Type {
    case "razorpay":
        return createRazorpayFromConfig(spec.Config)
    case "stripe":
        return createStripeFromConfig(spec.Config)
    }

    return nil
}
```

### 2. Dependency Injection Integration

```go
type Container struct {
    factories map[string]PaymentGatewayFactory
    services  map[string]*PaymentService
}

func (c *Container) RegisterFactory(name string, factory PaymentGatewayFactory) {
    c.factories[name] = factory
    c.services[name] = NewPaymentService(factory)
}

func (c *Container) GetService(name string) (*PaymentService, error) {
    service, exists := c.services[name]
    if !exists {
        return nil, fmt.Errorf("service not found: %s", name)
    }
    return service, nil
}
```

### 3. Plugin System Integration

```go
type PluginManager struct {
    registry *FactoryRegistry
    plugins  map[string]Plugin
}

type Plugin interface {
    Name() string
    CreateFactory() PaymentGatewayFactory
    Load() error
    Unload() error
}

func (p *PluginManager) LoadPlugin(pluginPath string) error {
    plugin, err := loadPlugin(pluginPath)
    if err != nil {
        return err
    }

    if err := plugin.Load(); err != nil {
        return err
    }

    p.plugins[plugin.Name()] = plugin
    p.registry.Register(plugin.Name(), plugin.CreateFactory())

    return nil
}
```

### 4. Circuit Breaker Integration

```go
type ResilientPaymentService struct {
    services        map[string]*PaymentService
    circuitBreakers map[string]*CircuitBreaker
    loadBalancer    LoadBalancer
}

func (r *ResilientPaymentService) ProcessTransaction(req TransactionRequest) error {
    // Select service based on load balancing
    serviceName := r.loadBalancer.SelectService(req.Region, req.Currency)

    // Get circuit breaker for service
    cb := r.circuitBreakers[serviceName]

    // Execute through circuit breaker
    return cb.Execute(func() error {
        service := r.services[serviceName]
        return service.ProcessTransaction(
            req.TransactionID,
            req.UserID,
            req.CardNumber,
            req.Amount,
            req.Currency,
        )
    })
}
```

## Common Interview Questions

### 1. **How does Abstract Factory differ from Factory Method?**

**Answer:**

- **Factory Method**: Creates single products with variants. One creator interface, multiple concrete creators, each creating one product type.
- **Abstract Factory**: Creates families of related products. One abstract factory interface, multiple concrete factories, each creating multiple related product types.

**Example:**

```go
// Factory Method - creates different payment processors
type ProcessorFactory interface {
    CreateProcessor() PaymentProcessor
}

// Abstract Factory - creates complete payment gateway families
type PaymentGatewayFactory interface {
    CreateProcessor() PaymentProcessor
    CreateValidator() PaymentValidator
    CreateLogger() PaymentLogger
}
```

### 2. **When would you choose Abstract Factory over Builder pattern?**

**Answer:**
Use **Abstract Factory** when you need to create families of related objects that must be used together.
Use **Builder** when you need to construct complex objects step by step with optional parameters.

**Abstract Factory**: "Create a family of payment gateway components (processor, validator, logger) that work together"
**Builder**: "Build a complex payment request with optional fields (metadata, webhooks, retries)"

### 3. **How do you handle the addition of new products to existing families?**

**Answer:**
This is a key limitation of Abstract Factory. Adding new products requires:

1. **Modify all factory interfaces** (breaks Open/Closed Principle)

```go
// Adding new product breaks existing factories
type PaymentGatewayFactory interface {
    CreateProcessor() PaymentProcessor
    CreateValidator() PaymentValidator
    CreateLogger() PaymentLogger
    CreateNotifier() PaymentNotifier // New product!
}
```

2. **Solutions:**
   - **Extension interfaces**: Create separate interfaces for new products
   - **Registry pattern**: Use product registries instead of fixed interfaces
   - **Plugin architecture**: Load products dynamically

### 4. **How do you ensure products from different families don't get mixed?**

**Answer:**

1. **Type System**: Use strong typing to prevent mixing

```go
type RazorpayProduct interface {
    GetGateway() string // Returns "razorpay"
}

type StripeProduct interface {
    GetGateway() string // Returns "stripe"
}
```

2. **Factory Ownership**: Services only accept products from their factory

```go
type PaymentService struct {
    factoryName string
    products    []interface{}
}

func (p *PaymentService) AddProduct(product interface{}) error {
    if gatewayProduct, ok := product.(GatewayProduct); ok {
        if gatewayProduct.GetGateway() != p.factoryName {
            return fmt.Errorf("product mismatch: expected %s, got %s",
                p.factoryName, gatewayProduct.GetGateway())
        }
    }
    p.products = append(p.products, product)
    return nil
}
```

3. **Factory Validation**: Validate product compatibility

```go
func (f *RazorpayFactory) ValidateProducts(products []interface{}) error {
    for _, product := range products {
        if !f.isCompatible(product) {
            return fmt.Errorf("incompatible product: %T", product)
        }
    }
    return nil
}
```

### 5. **How do you implement versioning in Abstract Factory?**

**Answer:**

1. **Versioned Factories**:

```go
type PaymentGatewayFactoryV1 interface {
    CreateProcessor() PaymentProcessor
    CreateValidator() PaymentValidator
}

type PaymentGatewayFactoryV2 interface {
    PaymentGatewayFactoryV1
    CreateNotifier() PaymentNotifier
    CreateReporter() PaymentReporter
}
```

2. **Version-aware Factory Selection**:

```go
type FactoryManager struct {
    factories map[string]map[string]PaymentGatewayFactory // [gateway][version]
}

func (f *FactoryManager) GetFactory(gateway, version string) PaymentGatewayFactory {
    if versions, exists := f.factories[gateway]; exists {
        if factory, exists := versions[version]; exists {
            return factory
        }
        // Fall back to latest version
        return versions["latest"]
    }
    return nil
}
```

3. **Backward Compatibility**:

```go
type BackwardCompatibleFactory struct {
    v1Factory PaymentGatewayFactoryV1
    v2Factory PaymentGatewayFactoryV2
}

func (b *BackwardCompatibleFactory) CreateProcessor() PaymentProcessor {
    if b.v2Factory != nil {
        return b.v2Factory.CreateProcessor()
    }
    return b.v1Factory.CreateProcessor()
}
```
