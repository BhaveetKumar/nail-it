# Factory Pattern

## Pattern Name & Intent

**Factory** - Create objects without specifying their exact classes.

The Factory pattern provides an interface for creating objects in a superclass, but allows subclasses to alter the type of objects that will be created. This pattern is particularly useful when you need to create objects based on runtime conditions or when you want to decouple object creation from object usage.

## When to Use

### Appropriate Scenarios
- **Runtime Object Creation**: When object type is determined at runtime
- **Complex Object Construction**: When object creation involves complex logic
- **Decoupling**: When you want to decouple object creation from usage
- **Multiple Implementations**: When you have multiple implementations of the same interface
- **Configuration-Based Creation**: When object type depends on configuration

### When NOT to Use
- **Simple Object Creation**: When object creation is straightforward
- **Performance Critical**: When factory overhead is too high
- **Single Implementation**: When you only have one implementation
- **Static Object Types**: When object types are known at compile time

## Real-World Use Cases (Fintech/Payments)

### Payment Gateway Factory
```go
// Payment gateway interface
type PaymentGateway interface {
    ProcessPayment(amount float64, currency string) (*PaymentResult, error)
    RefundPayment(transactionID string, amount float64) (*RefundResult, error)
    GetPaymentStatus(transactionID string) (*PaymentStatus, error)
}

// Payment result structures
type PaymentResult struct {
    TransactionID string `json:"transaction_id"`
    Status        string `json:"status"`
    Amount        float64 `json:"amount"`
    Currency      string `json:"currency"`
    Gateway       string `json:"gateway"`
}

type RefundResult struct {
    RefundID      string `json:"refund_id"`
    Status        string `json:"status"`
    Amount        float64 `json:"amount"`
    TransactionID string `json:"transaction_id"`
}

type PaymentStatus struct {
    TransactionID string `json:"transaction_id"`
    Status        string `json:"status"`
    Amount        float64 `json:"amount"`
    CreatedAt     time.Time `json:"created_at"`
}

// Concrete payment gateways
type StripeGateway struct {
    apiKey string
}

func NewStripeGateway(apiKey string) *StripeGateway {
    return &StripeGateway{apiKey: apiKey}
}

func (sg *StripeGateway) ProcessPayment(amount float64, currency string) (*PaymentResult, error) {
    // Simulate Stripe payment processing
    return &PaymentResult{
        TransactionID: "stripe_" + generateID(),
        Status:        "success",
        Amount:        amount,
        Currency:      currency,
        Gateway:       "stripe",
    }, nil
}

func (sg *StripeGateway) RefundPayment(transactionID string, amount float64) (*RefundResult, error) {
    return &RefundResult{
        RefundID:      "refund_" + generateID(),
        Status:        "success",
        Amount:        amount,
        TransactionID: transactionID,
    }, nil
}

func (sg *StripeGateway) GetPaymentStatus(transactionID string) (*PaymentStatus, error) {
    return &PaymentStatus{
        TransactionID: transactionID,
        Status:        "completed",
        Amount:        100.50,
        CreatedAt:     time.Now(),
    }, nil
}

type PayPalGateway struct {
    clientID     string
    clientSecret string
}

func NewPayPalGateway(clientID, clientSecret string) *PayPalGateway {
    return &PayPalGateway{
        clientID:     clientID,
        clientSecret: clientSecret,
    }
}

func (pg *PayPalGateway) ProcessPayment(amount float64, currency string) (*PaymentResult, error) {
    // Simulate PayPal payment processing
    return &PaymentResult{
        TransactionID: "paypal_" + generateID(),
        Status:        "success",
        Amount:        amount,
        Currency:      currency,
        Gateway:       "paypal",
    }, nil
}

func (pg *PayPalGateway) RefundPayment(transactionID string, amount float64) (*RefundResult, error) {
    return &RefundResult{
        RefundID:      "refund_" + generateID(),
        Status:        "success",
        Amount:        amount,
        TransactionID: transactionID,
    }, nil
}

func (pg *PayPalGateway) GetPaymentStatus(transactionID string) (*PaymentStatus, error) {
    return &PaymentStatus{
        TransactionID: transactionID,
        Status:        "completed",
        Amount:        100.50,
        CreatedAt:     time.Now(),
    }, nil
}

// Payment gateway factory
type PaymentGatewayFactory struct {
    configs map[string]GatewayConfig
}

type GatewayConfig struct {
    Type         string
    APIKey       string
    ClientID     string
    ClientSecret string
    Environment  string
}

func NewPaymentGatewayFactory() *PaymentGatewayFactory {
    return &PaymentGatewayFactory{
        configs: make(map[string]GatewayConfig),
    }
}

func (pgf *PaymentGatewayFactory) RegisterGateway(name string, config GatewayConfig) {
    pgf.configs[name] = config
}

func (pgf *PaymentGatewayFactory) CreateGateway(gatewayType string) (PaymentGateway, error) {
    config, exists := pgf.configs[gatewayType]
    if !exists {
        return nil, fmt.Errorf("gateway type %s not supported", gatewayType)
    }

    switch config.Type {
    case "stripe":
        return NewStripeGateway(config.APIKey), nil
    case "paypal":
        return NewPayPalGateway(config.ClientID, config.ClientSecret), nil
    default:
        return nil, fmt.Errorf("unsupported gateway type: %s", config.Type)
    }
}
```

### Notification Channel Factory
```go
// Notification channel interface
type NotificationChannel interface {
    Send(recipient string, message string) error
    GetChannelType() string
}

// Email notification channel
type EmailChannel struct {
    smtpHost string
    smtpPort int
    username string
    password string
}

func NewEmailChannel(smtpHost string, smtpPort int, username, password string) *EmailChannel {
    return &EmailChannel{
        smtpHost: smtpHost,
        smtpPort: smtpPort,
        username: username,
        password: password,
    }
}

func (ec *EmailChannel) Send(recipient string, message string) error {
    // Simulate email sending
    log.Printf("Sending email to %s: %s", recipient, message)
    return nil
}

func (ec *EmailChannel) GetChannelType() string {
    return "email"
}

// SMS notification channel
type SMSChannel struct {
    apiKey    string
    apiSecret string
}

func NewSMSChannel(apiKey, apiSecret string) *SMSChannel {
    return &SMSChannel{
        apiKey:    apiKey,
        apiSecret: apiSecret,
    }
}

func (sc *SMSChannel) Send(recipient string, message string) error {
    // Simulate SMS sending
    log.Printf("Sending SMS to %s: %s", recipient, message)
    return nil
}

func (sc *SMSChannel) GetChannelType() string {
    return "sms"
}

// Push notification channel
type PushChannel struct {
    serverKey string
}

func NewPushChannel(serverKey string) *PushChannel {
    return &PushChannel{serverKey: serverKey}
}

func (pc *PushChannel) Send(recipient string, message string) error {
    // Simulate push notification
    log.Printf("Sending push notification to %s: %s", recipient, message)
    return nil
}

func (pc *PushChannel) GetChannelType() string {
    return "push"
}

// Notification channel factory
type NotificationChannelFactory struct {
    configs map[string]ChannelConfig
}

type ChannelConfig struct {
    Type       string
    SMTPHost   string
    SMTPPort   int
    Username   string
    Password   string
    APIKey     string
    APISecret  string
    ServerKey  string
}

func NewNotificationChannelFactory() *NotificationChannelFactory {
    return &NotificationChannelFactory{
        configs: make(map[string]ChannelConfig),
    }
}

func (ncf *NotificationChannelFactory) RegisterChannel(name string, config ChannelConfig) {
    ncf.configs[name] = config
}

func (ncf *NotificationChannelFactory) CreateChannel(channelType string) (NotificationChannel, error) {
    config, exists := ncf.configs[channelType]
    if !exists {
        return nil, fmt.Errorf("channel type %s not supported", channelType)
    }

    switch config.Type {
    case "email":
        return NewEmailChannel(config.SMTPHost, config.SMTPPort, config.Username, config.Password), nil
    case "sms":
        return NewSMSChannel(config.APIKey, config.APISecret), nil
    case "push":
        return NewPushChannel(config.ServerKey), nil
    default:
        return nil, fmt.Errorf("unsupported channel type: %s", config.Type)
    }
}
```

## Go Implementation

### Generic Factory Pattern
```go
package main

import (
    "fmt"
    "sync"
)

// Generic factory interface
type Factory[T any] interface {
    Create(config interface{}) (T, error)
    RegisterType(name string, creator func(interface{}) (T, error))
}

// Generic factory implementation
type GenericFactory[T any] struct {
    creators map[string]func(interface{}) (T, error)
    mutex    sync.RWMutex
}

func NewGenericFactory[T any]() *GenericFactory[T] {
    return &GenericFactory[T]{
        creators: make(map[string]func(interface{}) (T, error)),
    }
}

func (gf *GenericFactory[T]) RegisterType(name string, creator func(interface{}) (T, error)) {
    gf.mutex.Lock()
    defer gf.mutex.Unlock()
    gf.creators[name] = creator
}

func (gf *GenericFactory[T]) Create(typeName string, config interface{}) (T, error) {
    gf.mutex.RLock()
    creator, exists := gf.creators[typeName]
    gf.mutex.RUnlock()

    if !exists {
        var zero T
        return zero, fmt.Errorf("type %s not registered", typeName)
    }

    return creator(config)
}

// Database connection factory
type DatabaseConnection interface {
    Connect() error
    Disconnect() error
    Query(sql string) (interface{}, error)
}

type PostgreSQLConnection struct {
    host     string
    port     int
    database string
    username string
    password string
}

func NewPostgreSQLConnection(config interface{}) (DatabaseConnection, error) {
    cfg, ok := config.(map[string]interface{})
    if !ok {
        return nil, fmt.Errorf("invalid config type")
    }

    return &PostgreSQLConnection{
        host:     cfg["host"].(string),
        port:     cfg["port"].(int),
        database: cfg["database"].(string),
        username: cfg["username"].(string),
        password: cfg["password"].(string),
    }, nil
}

func (pgc *PostgreSQLConnection) Connect() error {
    log.Printf("Connecting to PostgreSQL: %s:%d/%s", pgc.host, pgc.port, pgc.database)
    return nil
}

func (pgc *PostgreSQLConnection) Disconnect() error {
    log.Printf("Disconnecting from PostgreSQL")
    return nil
}

func (pgc *PostgreSQLConnection) Query(sql string) (interface{}, error) {
    log.Printf("Executing query: %s", sql)
    return "query result", nil
}

type MySQLConnection struct {
    host     string
    port     int
    database string
    username string
    password string
}

func NewMySQLConnection(config interface{}) (DatabaseConnection, error) {
    cfg, ok := config.(map[string]interface{})
    if !ok {
        return nil, fmt.Errorf("invalid config type")
    }

    return &MySQLConnection{
        host:     cfg["host"].(string),
        port:     cfg["port"].(int),
        database: cfg["database"].(string),
        username: cfg["username"].(string),
        password: cfg["password"].(string),
    }, nil
}

func (mc *MySQLConnection) Connect() error {
    log.Printf("Connecting to MySQL: %s:%d/%s", mc.host, mc.port, mc.database)
    return nil
}

func (mc *MySQLConnection) Disconnect() error {
    log.Printf("Disconnecting from MySQL")
    return nil
}

func (mc *MySQLConnection) Query(sql string) (interface{}, error) {
    log.Printf("Executing query: %s", sql)
    return "query result", nil
}
```

### Abstract Factory Pattern
```go
// Abstract factory for UI components
type UIComponent interface {
    Render() string
    GetType() string
}

type Button interface {
    UIComponent
    Click() string
}

type TextField interface {
    UIComponent
    SetText(text string)
    GetText() string
}

// Web UI components
type WebButton struct {
    text string
}

func (wb *WebButton) Render() string {
    return fmt.Sprintf("<button>%s</button>", wb.text)
}

func (wb *WebButton) GetType() string {
    return "web_button"
}

func (wb *WebButton) Click() string {
    return "Web button clicked"
}

type WebTextField struct {
    text string
}

func (wtf *WebTextField) Render() string {
    return fmt.Sprintf("<input type='text' value='%s'>", wtf.text)
}

func (wtf *WebTextField) GetType() string {
    return "web_textfield"
}

func (wtf *WebTextField) SetText(text string) {
    wtf.text = text
}

func (wtf *WebTextField) GetText() string {
    return wtf.text
}

// Mobile UI components
type MobileButton struct {
    text string
}

func (mb *MobileButton) Render() string {
    return fmt.Sprintf("MobileButton: %s", mb.text)
}

func (mb *MobileButton) GetType() string {
    return "mobile_button"
}

func (mb *MobileButton) Click() string {
    return "Mobile button tapped"
}

type MobileTextField struct {
    text string
}

func (mtf *MobileTextField) Render() string {
    return fmt.Sprintf("MobileTextField: %s", mtf.text)
}

func (mtf *MobileTextField) GetType() string {
    return "mobile_textfield"
}

func (mtf *MobileTextField) SetText(text string) {
    mtf.text = text
}

func (mtf *MobileTextField) GetText() string {
    return mtf.text
}

// Abstract factory interface
type UIFactory interface {
    CreateButton(text string) Button
    CreateTextField(placeholder string) TextField
}

// Web UI factory
type WebUIFactory struct{}

func (wuf *WebUIFactory) CreateButton(text string) Button {
    return &WebButton{text: text}
}

func (wuf *WebUIFactory) CreateTextField(placeholder string) TextField {
    return &WebTextField{text: placeholder}
}

// Mobile UI factory
type MobileUIFactory struct{}

func (muf *MobileUIFactory) CreateButton(text string) Button {
    return &MobileButton{text: text}
}

func (muf *MobileUIFactory) CreateTextField(placeholder string) TextField {
    return &MobileTextField{text: placeholder}
}

// Factory provider
type UIFactoryProvider struct {
    factories map[string]UIFactory
}

func NewUIFactoryProvider() *UIFactoryProvider {
    return &UIFactoryProvider{
        factories: map[string]UIFactory{
            "web":    &WebUIFactory{},
            "mobile": &MobileUIFactory{},
        },
    }
}

func (uip *UIFactoryProvider) GetFactory(platform string) (UIFactory, error) {
    factory, exists := uip.factories[platform]
    if !exists {
        return nil, fmt.Errorf("unsupported platform: %s", platform)
    }
    return factory, nil
}
```

## Variants & Trade-offs

### Variants

#### 1. Simple Factory
```go
type SimpleFactory struct{}

func (sf *SimpleFactory) CreatePaymentGateway(gatewayType string) PaymentGateway {
    switch gatewayType {
    case "stripe":
        return NewStripeGateway("stripe_key")
    case "paypal":
        return NewPayPalGateway("paypal_id", "paypal_secret")
    default:
        return nil
    }
}
```

**Pros**: Simple and straightforward
**Cons**: Violates Open/Closed principle, hard to extend

#### 2. Factory Method
```go
type PaymentGatewayCreator interface {
    CreateGateway() PaymentGateway
}

type StripeGatewayCreator struct{}

func (sgc *StripeGatewayCreator) CreateGateway() PaymentGateway {
    return NewStripeGateway("stripe_key")
}

type PayPalGatewayCreator struct{}

func (pgc *PayPalGatewayCreator) CreateGateway() PaymentGateway {
    return NewPayPalGateway("paypal_id", "paypal_secret")
}
```

**Pros**: Follows Open/Closed principle, easy to extend
**Cons**: More complex, requires more classes

#### 3. Abstract Factory
```go
type PaymentSystemFactory interface {
    CreateGateway() PaymentGateway
    CreateValidator() PaymentValidator
    CreateLogger() PaymentLogger
}
```

**Pros**: Creates families of related objects, high cohesion
**Cons**: Complex, hard to extend with new product types

### Trade-offs

| Aspect              | Pros                              | Cons                              |
| ------------------- | --------------------------------- | --------------------------------- |
| **Flexibility**     | Easy to add new types            | Can become complex               |
| **Decoupling**      | Separates creation from usage    | Additional abstraction layer     |
| **Testing**         | Easy to mock and test            | More interfaces to maintain      |
| **Performance**     | Lazy creation possible           | Factory overhead                 |
| **Maintainability** | Centralized object creation      | Can become over-engineered       |

## Testable Example

```go
package main

import (
    "testing"
)

// Mock payment gateway for testing
type MockPaymentGateway struct {
    shouldFail bool
    calls      []PaymentCall
}

type PaymentCall struct {
    Method   string
    Amount   float64
    Currency string
}

func NewMockPaymentGateway(shouldFail bool) *MockPaymentGateway {
    return &MockPaymentGateway{
        shouldFail: shouldFail,
        calls:      make([]PaymentCall, 0),
    }
}

func (mpg *MockPaymentGateway) ProcessPayment(amount float64, currency string) (*PaymentResult, error) {
    mpg.calls = append(mpg.calls, PaymentCall{
        Method:   "ProcessPayment",
        Amount:   amount,
        Currency: currency,
    })

    if mpg.shouldFail {
        return nil, fmt.Errorf("payment failed")
    }

    return &PaymentResult{
        TransactionID: "mock_" + generateID(),
        Status:        "success",
        Amount:        amount,
        Currency:      currency,
        Gateway:       "mock",
    }, nil
}

func (mpg *MockPaymentGateway) RefundPayment(transactionID string, amount float64) (*RefundResult, error) {
    mpg.calls = append(mpg.calls, PaymentCall{
        Method: "RefundPayment",
        Amount: amount,
    })

    if mpg.shouldFail {
        return nil, fmt.Errorf("refund failed")
    }

    return &RefundResult{
        RefundID:      "refund_" + generateID(),
        Status:        "success",
        Amount:        amount,
        TransactionID: transactionID,
    }, nil
}

func (mpg *MockPaymentGateway) GetPaymentStatus(transactionID string) (*PaymentStatus, error) {
    mpg.calls = append(mpg.calls, PaymentCall{
        Method: "GetPaymentStatus",
    })

    return &PaymentStatus{
        TransactionID: transactionID,
        Status:        "completed",
        Amount:        100.50,
        CreatedAt:     time.Now(),
    }, nil
}

func (mpg *MockPaymentGateway) GetCalls() []PaymentCall {
    return mpg.calls
}

// Tests
func TestPaymentGatewayFactory_CreateStripe(t *testing.T) {
    factory := NewPaymentGatewayFactory()
    factory.RegisterGateway("stripe", GatewayConfig{
        Type:    "stripe",
        APIKey:  "test_key",
    })

    gateway, err := factory.CreateGateway("stripe")
    if err != nil {
        t.Fatalf("CreateGateway() error = %v", err)
    }

    if gateway == nil {
        t.Error("Expected gateway to be created")
    }

    // Test payment processing
    result, err := gateway.ProcessPayment(100.50, "USD")
    if err != nil {
        t.Fatalf("ProcessPayment() error = %v", err)
    }

    if result.Gateway != "stripe" {
        t.Errorf("Expected gateway 'stripe', got %s", result.Gateway)
    }
}

func TestPaymentGatewayFactory_CreatePayPal(t *testing.T) {
    factory := NewPaymentGatewayFactory()
    factory.RegisterGateway("paypal", GatewayConfig{
        Type:         "paypal",
        ClientID:     "test_client_id",
        ClientSecret: "test_client_secret",
    })

    gateway, err := factory.CreateGateway("paypal")
    if err != nil {
        t.Fatalf("CreateGateway() error = %v", err)
    }

    if gateway == nil {
        t.Error("Expected gateway to be created")
    }

    // Test payment processing
    result, err := gateway.ProcessPayment(200.75, "EUR")
    if err != nil {
        t.Fatalf("ProcessPayment() error = %v", err)
    }

    if result.Gateway != "paypal" {
        t.Errorf("Expected gateway 'paypal', got %s", result.Gateway)
    }
}

func TestPaymentGatewayFactory_UnsupportedGateway(t *testing.T) {
    factory := NewPaymentGatewayFactory()

    gateway, err := factory.CreateGateway("unsupported")
    if err == nil {
        t.Error("Expected error for unsupported gateway")
    }

    if gateway != nil {
        t.Error("Expected gateway to be nil")
    }
}

func TestGenericFactory(t *testing.T) {
    factory := NewGenericFactory[DatabaseConnection]()
    factory.RegisterType("postgresql", NewPostgreSQLConnection)
    factory.RegisterType("mysql", NewMySQLConnection)

    // Test PostgreSQL connection
    config := map[string]interface{}{
        "host":     "localhost",
        "port":     5432,
        "database": "testdb",
        "username": "testuser",
        "password": "testpass",
    }

    conn, err := factory.Create("postgresql", config)
    if err != nil {
        t.Fatalf("Create() error = %v", err)
    }

    if conn == nil {
        t.Error("Expected connection to be created")
    }

    // Test connection
    err = conn.Connect()
    if err != nil {
        t.Fatalf("Connect() error = %v", err)
    }
}

func TestAbstractFactory(t *testing.T) {
    provider := NewUIFactoryProvider()

    // Test web factory
    webFactory, err := provider.GetFactory("web")
    if err != nil {
        t.Fatalf("GetFactory() error = %v", err)
    }

    button := webFactory.CreateButton("Click Me")
    if button.GetType() != "web_button" {
        t.Errorf("Expected 'web_button', got %s", button.GetType())
    }

    textField := webFactory.CreateTextField("Enter text")
    if textField.GetType() != "web_textfield" {
        t.Errorf("Expected 'web_textfield', got %s", textField.GetType())
    }

    // Test mobile factory
    mobileFactory, err := provider.GetFactory("mobile")
    if err != nil {
        t.Fatalf("GetFactory() error = %v", err)
    }

    mobileButton := mobileFactory.CreateButton("Tap Me")
    if mobileButton.GetType() != "mobile_button" {
        t.Errorf("Expected 'mobile_button', got %s", mobileButton.GetType())
    }
}
```

## Integration Tips

### 1. With Dependency Injection
```go
type PaymentService struct {
    gatewayFactory *PaymentGatewayFactory
}

func NewPaymentService(gatewayFactory *PaymentGatewayFactory) *PaymentService {
    return &PaymentService{
        gatewayFactory: gatewayFactory,
    }
}

func (ps *PaymentService) ProcessPayment(gatewayType string, amount float64, currency string) error {
    gateway, err := ps.gatewayFactory.CreateGateway(gatewayType)
    if err != nil {
        return err
    }

    _, err = gateway.ProcessPayment(amount, currency)
    return err
}
```

### 2. With Configuration
```go
type FactoryConfig struct {
    Gateways map[string]GatewayConfig `yaml:"gateways"`
}

func LoadFactoryFromConfig(configPath string) (*PaymentGatewayFactory, error) {
    var config FactoryConfig
    data, err := os.ReadFile(configPath)
    if err != nil {
        return nil, err
    }

    if err := yaml.Unmarshal(data, &config); err != nil {
        return nil, err
    }

    factory := NewPaymentGatewayFactory()
    for name, gatewayConfig := range config.Gateways {
        factory.RegisterGateway(name, gatewayConfig)
    }

    return factory, nil
}
```

### 3. With Builder Pattern
```go
type PaymentGatewayBuilder struct {
    factory *PaymentGatewayFactory
}

func NewPaymentGatewayBuilder() *PaymentGatewayBuilder {
    return &PaymentGatewayBuilder{
        factory: NewPaymentGatewayFactory(),
    }
}

func (pgb *PaymentGatewayBuilder) WithStripe(apiKey string) *PaymentGatewayBuilder {
    pgb.factory.RegisterGateway("stripe", GatewayConfig{
        Type:    "stripe",
        APIKey:  apiKey,
    })
    return pgb
}

func (pgb *PaymentGatewayBuilder) WithPayPal(clientID, clientSecret string) *PaymentGatewayBuilder {
    pgb.factory.RegisterGateway("paypal", GatewayConfig{
        Type:         "paypal",
        ClientID:     clientID,
        ClientSecret: clientSecret,
    })
    return pgb
}

func (pgb *PaymentGatewayBuilder) Build() *PaymentGatewayFactory {
    return pgb.factory
}
```

## Common Interview Questions

### 1. What is the Factory pattern and when would you use it?

**Answer**: The Factory pattern provides an interface for creating objects without specifying their exact classes. Use it when object creation is complex, when you need to create objects based on runtime conditions, or when you want to decouple object creation from object usage.

### 2. What's the difference between Factory Method and Abstract Factory?

**Answer**: Factory Method creates objects of a single type, while Abstract Factory creates families of related objects. Factory Method is simpler and focuses on one product, while Abstract Factory is more complex and creates multiple related products.

### 3. How do you implement the Factory pattern in Go?

**Answer**: Define an interface for the products, create concrete implementations, implement a factory that returns the appropriate concrete type based on input parameters, and use dependency injection to provide the factory to consumers.

### 4. What are the benefits and drawbacks of the Factory pattern?

**Answer**: Benefits include decoupling object creation from usage, easy addition of new types, and centralized creation logic. Drawbacks include additional complexity, potential over-engineering, and factory overhead.

### 5. How do you make a Factory pattern testable?

**Answer**: Use dependency injection to provide the factory, create mock implementations for testing, use interfaces for all dependencies, and implement factory methods that can return different implementations based on configuration or environment.
