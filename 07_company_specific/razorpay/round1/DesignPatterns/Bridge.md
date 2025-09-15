# Bridge Pattern

## Pattern Name & Intent

**Bridge** is a structural design pattern that separates an abstraction from its implementation so that the two can vary independently. It uses composition instead of inheritance to connect different hierarchies.

**Key Intent:**

- Decouple abstraction from implementation
- Allow both abstraction and implementation to vary independently
- Support multiple implementations for a single abstraction
- Enable runtime switching between implementations
- Avoid permanent binding between abstraction and implementation

## When to Use

**Use Bridge when:**

1. **Multiple Implementations**: Want to share implementation among multiple objects
2. **Runtime Selection**: Need to select implementation at runtime
3. **Independent Variation**: Both abstraction and implementation should vary independently
4. **Platform Independence**: Want to hide implementation details from clients
5. **Inheritance Explosion**: Want to avoid combinatorial explosion of classes
6. **Compilation Dependencies**: Want to reduce compile-time dependencies

**Don't use when:**

- Only one implementation exists
- Abstraction and implementation are tightly coupled
- Simple hierarchy without multiple implementations
- Performance is critical (adds indirection)

## Real-World Use Cases (Payments/Fintech)

### 1. Payment Method Bridge

```go
// Payment methods (abstraction) can use different processors (implementation)
type PaymentMethod interface {
    ProcessPayment(amount decimal.Decimal) (*PaymentResult, error)
    ValidatePayment() error
    GetPaymentDetails() *PaymentDetails
}

// Different payment processors (implementations)
type PaymentProcessor interface {
    Process(req ProcessRequest) (*ProcessResult, error)
    Validate(req ValidateRequest) error
    GetCapabilities() *Capabilities
}

// Credit card payment using different processors
type CreditCardPayment struct {
    processor PaymentProcessor
    cardInfo  *CreditCardInfo
}

func (c *CreditCardPayment) ProcessPayment(amount decimal.Decimal) (*PaymentResult, error) {
    req := ProcessRequest{
        Type:   "credit_card",
        Amount: amount,
        Data:   c.cardInfo,
    }

    result, err := c.processor.Process(req)
    if err != nil {
        return nil, err
    }

    return &PaymentResult{
        TransactionID: result.ID,
        Status:       result.Status,
        Amount:       amount,
    }, nil
}

// UPI payment using different processors
type UPIPayment struct {
    processor PaymentProcessor
    upiID     string
}

func (u *UPIPayment) ProcessPayment(amount decimal.Decimal) (*PaymentResult, error) {
    req := ProcessRequest{
        Type:   "upi",
        Amount: amount,
        Data:   map[string]interface{}{"upi_id": u.upiID},
    }

    return u.processor.Process(req)
}
```

### 2. Notification Bridge

```go
// Different notification types (abstraction)
type Notification interface {
    Send() error
    Schedule(time time.Time) error
    GetStatus() NotificationStatus
}

// Different delivery channels (implementation)
type NotificationChannel interface {
    Deliver(message string, recipient string) error
    GetDeliveryStatus(id string) DeliveryStatus
    GetCapabilities() ChannelCapabilities
}

// SMS notification using different channels
type SMSNotification struct {
    channel   NotificationChannel
    message   string
    recipient string
}

func (s *SMSNotification) Send() error {
    return s.channel.Deliver(s.message, s.recipient)
}

// Email notification using different channels
type EmailNotification struct {
    channel   NotificationChannel
    subject   string
    body      string
    recipient string
}

func (e *EmailNotification) Send() error {
    message := fmt.Sprintf("Subject: %s\n\n%s", e.subject, e.body)
    return e.channel.Deliver(message, e.recipient)
}
```

### 3. Database Bridge

```go
// Different data access patterns (abstraction)
type Repository interface {
    Save(entity interface{}) error
    FindByID(id string) (interface{}, error)
    FindAll() ([]interface{}, error)
    Delete(id string) error
}

// Different storage implementations (implementation)
type Storage interface {
    Create(table string, data map[string]interface{}) error
    Read(table string, id string) (map[string]interface{}, error)
    Update(table string, id string, data map[string]interface{}) error
    Delete(table string, id string) error
}

// User repository using different storage backends
type UserRepository struct {
    storage Storage
    table   string
}

func (u *UserRepository) Save(entity interface{}) error {
    user := entity.(*User)
    data := map[string]interface{}{
        "id":    user.ID,
        "name":  user.Name,
        "email": user.Email,
    }
    return u.storage.Create(u.table, data)
}

// Transaction repository using different storage backends
type TransactionRepository struct {
    storage Storage
    table   string
}

func (t *TransactionRepository) Save(entity interface{}) error {
    transaction := entity.(*Transaction)
    data := map[string]interface{}{
        "id":     transaction.ID,
        "amount": transaction.Amount,
        "status": transaction.Status,
    }
    return t.storage.Create(t.table, data)
}
```

### 4. Trading Platform Bridge

```go
// Different order types (abstraction)
type Order interface {
    Execute() (*ExecutionResult, error)
    Validate() error
    GetOrderInfo() *OrderInfo
}

// Different execution venues (implementation)
type ExecutionVenue interface {
    ExecuteOrder(order OrderDetails) (*VenueResult, error)
    GetMarketData(symbol string) (*MarketData, error)
    GetVenueInfo() *VenueInfo
}

// Market order using different venues
type MarketOrder struct {
    venue    ExecutionVenue
    symbol   string
    quantity int64
    side     OrderSide
}

func (m *MarketOrder) Execute() (*ExecutionResult, error) {
    orderDetails := OrderDetails{
        Type:     "market",
        Symbol:   m.symbol,
        Quantity: m.quantity,
        Side:     m.side,
    }

    result, err := m.venue.ExecuteOrder(orderDetails)
    if err != nil {
        return nil, err
    }

    return &ExecutionResult{
        OrderID:       result.ID,
        ExecutedPrice: result.Price,
        ExecutedQty:   result.Quantity,
        Status:        result.Status,
    }, nil
}

// Limit order using different venues
type LimitOrder struct {
    venue      ExecutionVenue
    symbol     string
    quantity   int64
    limitPrice decimal.Decimal
    side       OrderSide
}

func (l *LimitOrder) Execute() (*ExecutionResult, error) {
    orderDetails := OrderDetails{
        Type:       "limit",
        Symbol:     l.symbol,
        Quantity:   l.quantity,
        LimitPrice: l.limitPrice,
        Side:       l.side,
    }

    result, err := l.venue.ExecuteOrder(orderDetails)
    if err != nil {
        return nil, err
    }

    return &ExecutionResult{
        OrderID:       result.ID,
        ExecutedPrice: result.Price,
        ExecutedQty:   result.Quantity,
        Status:        result.Status,
    }, nil
}
```

## Go Implementation

```go
package main

import (
    "fmt"
    "log"
    "time"
    "github.com/shopspring/decimal"
)

// Abstraction: Payment Methods
type PaymentMethod interface {
    ProcessPayment(amount decimal.Decimal, currency string) (*PaymentResult, error)
    ValidatePayment() error
    GetPaymentDetails() *PaymentDetails
    SetProcessor(processor PaymentProcessor)
}

// Implementation: Payment Processors
type PaymentProcessor interface {
    ProcessTransaction(req *TransactionRequest) (*TransactionResult, error)
    ValidateTransaction(req *ValidationRequest) error
    GetProcessorInfo() *ProcessorInfo
}

// Common data structures
type PaymentResult struct {
    TransactionID string          `json:"transaction_id"`
    Status        string          `json:"status"`
    Amount        decimal.Decimal `json:"amount"`
    Currency      string          `json:"currency"`
    ProcessedAt   time.Time       `json:"processed_at"`
    Fee           decimal.Decimal `json:"fee"`
}

type PaymentDetails struct {
    PaymentType string                 `json:"payment_type"`
    Details     map[string]interface{} `json:"details"`
    IsValid     bool                   `json:"is_valid"`
}

type TransactionRequest struct {
    PaymentType string                 `json:"payment_type"`
    Amount      decimal.Decimal        `json:"amount"`
    Currency    string                 `json:"currency"`
    PaymentData map[string]interface{} `json:"payment_data"`
    Metadata    map[string]string      `json:"metadata"`
}

type TransactionResult struct {
    ID          string          `json:"id"`
    Status      string          `json:"status"`
    Amount      decimal.Decimal `json:"amount"`
    Fee         decimal.Decimal `json:"fee"`
    ProcessedAt time.Time       `json:"processed_at"`
    Reference   string          `json:"reference"`
}

type ValidationRequest struct {
    PaymentType string                 `json:"payment_type"`
    PaymentData map[string]interface{} `json:"payment_data"`
}

type ProcessorInfo struct {
    Name         string   `json:"name"`
    Version      string   `json:"version"`
    Capabilities []string `json:"capabilities"`
    SupportedCurrencies []string `json:"supported_currencies"`
}

// Concrete Abstraction 1: Credit Card Payment
type CreditCardPayment struct {
    processor  PaymentProcessor
    cardNumber string
    expiryDate string
    cvv        string
    holderName string
}

func NewCreditCardPayment(cardNumber, expiryDate, cvv, holderName string) *CreditCardPayment {
    return &CreditCardPayment{
        cardNumber: cardNumber,
        expiryDate: expiryDate,
        cvv:        cvv,
        holderName: holderName,
    }
}

func (c *CreditCardPayment) SetProcessor(processor PaymentProcessor) {
    c.processor = processor
}

func (c *CreditCardPayment) ProcessPayment(amount decimal.Decimal, currency string) (*PaymentResult, error) {
    if c.processor == nil {
        return nil, fmt.Errorf("no payment processor set")
    }

    // Validate payment first
    if err := c.ValidatePayment(); err != nil {
        return nil, fmt.Errorf("payment validation failed: %w", err)
    }

    // Prepare transaction request
    req := &TransactionRequest{
        PaymentType: "credit_card",
        Amount:      amount,
        Currency:    currency,
        PaymentData: map[string]interface{}{
            "card_number":  c.maskCardNumber(c.cardNumber),
            "expiry_date":  c.expiryDate,
            "holder_name":  c.holderName,
            "card_type":    c.detectCardType(c.cardNumber),
        },
        Metadata: map[string]string{
            "payment_method": "credit_card",
            "processor":      c.processor.GetProcessorInfo().Name,
        },
    }

    // Process through the bridge
    result, err := c.processor.ProcessTransaction(req)
    if err != nil {
        return nil, fmt.Errorf("transaction processing failed: %w", err)
    }

    // Convert processor result to payment result
    return &PaymentResult{
        TransactionID: result.ID,
        Status:        result.Status,
        Amount:        result.Amount,
        Currency:      currency,
        ProcessedAt:   result.ProcessedAt,
        Fee:           result.Fee,
    }, nil
}

func (c *CreditCardPayment) ValidatePayment() error {
    if c.processor == nil {
        return fmt.Errorf("no payment processor set")
    }

    req := &ValidationRequest{
        PaymentType: "credit_card",
        PaymentData: map[string]interface{}{
            "card_number": c.cardNumber,
            "expiry_date": c.expiryDate,
            "cvv":         c.cvv,
        },
    }

    return c.processor.ValidateTransaction(req)
}

func (c *CreditCardPayment) GetPaymentDetails() *PaymentDetails {
    return &PaymentDetails{
        PaymentType: "credit_card",
        Details: map[string]interface{}{
            "masked_card_number": c.maskCardNumber(c.cardNumber),
            "card_type":          c.detectCardType(c.cardNumber),
            "holder_name":        c.holderName,
            "expiry_date":        c.expiryDate,
        },
        IsValid: c.ValidatePayment() == nil,
    }
}

func (c *CreditCardPayment) maskCardNumber(cardNumber string) string {
    if len(cardNumber) < 4 {
        return "****"
    }
    return "****-****-****-" + cardNumber[len(cardNumber)-4:]
}

func (c *CreditCardPayment) detectCardType(cardNumber string) string {
    if len(cardNumber) >= 2 {
        prefix := cardNumber[:2]
        switch {
        case prefix == "40" || prefix == "41" || prefix == "42" || prefix == "43":
            return "visa"
        case prefix == "51" || prefix == "52" || prefix == "53" || prefix == "54" || prefix == "55":
            return "mastercard"
        case prefix == "34" || prefix == "37":
            return "amex"
        }
    }
    return "unknown"
}

// Concrete Abstraction 2: Digital Wallet Payment
type DigitalWalletPayment struct {
    processor   PaymentProcessor
    walletID    string
    walletType  string
    phoneNumber string
}

func NewDigitalWalletPayment(walletID, walletType, phoneNumber string) *DigitalWalletPayment {
    return &DigitalWalletPayment{
        walletID:    walletID,
        walletType:  walletType,
        phoneNumber: phoneNumber,
    }
}

func (d *DigitalWalletPayment) SetProcessor(processor PaymentProcessor) {
    d.processor = processor
}

func (d *DigitalWalletPayment) ProcessPayment(amount decimal.Decimal, currency string) (*PaymentResult, error) {
    if d.processor == nil {
        return nil, fmt.Errorf("no payment processor set")
    }

    // Validate payment first
    if err := d.ValidatePayment(); err != nil {
        return nil, fmt.Errorf("payment validation failed: %w", err)
    }

    // Prepare transaction request
    req := &TransactionRequest{
        PaymentType: "digital_wallet",
        Amount:      amount,
        Currency:    currency,
        PaymentData: map[string]interface{}{
            "wallet_id":     d.walletID,
            "wallet_type":   d.walletType,
            "phone_number":  d.phoneNumber,
        },
        Metadata: map[string]string{
            "payment_method": "digital_wallet",
            "wallet_type":    d.walletType,
            "processor":      d.processor.GetProcessorInfo().Name,
        },
    }

    // Process through the bridge
    result, err := d.processor.ProcessTransaction(req)
    if err != nil {
        return nil, fmt.Errorf("transaction processing failed: %w", err)
    }

    // Convert processor result to payment result
    return &PaymentResult{
        TransactionID: result.ID,
        Status:        result.Status,
        Amount:        result.Amount,
        Currency:      currency,
        ProcessedAt:   result.ProcessedAt,
        Fee:           result.Fee,
    }, nil
}

func (d *DigitalWalletPayment) ValidatePayment() error {
    if d.processor == nil {
        return fmt.Errorf("no payment processor set")
    }

    req := &ValidationRequest{
        PaymentType: "digital_wallet",
        PaymentData: map[string]interface{}{
            "wallet_id":    d.walletID,
            "wallet_type":  d.walletType,
            "phone_number": d.phoneNumber,
        },
    }

    return d.processor.ValidateTransaction(req)
}

func (d *DigitalWalletPayment) GetPaymentDetails() *PaymentDetails {
    return &PaymentDetails{
        PaymentType: "digital_wallet",
        Details: map[string]interface{}{
            "wallet_id":    d.walletID,
            "wallet_type":  d.walletType,
            "phone_number": d.phoneNumber,
        },
        IsValid: d.ValidatePayment() == nil,
    }
}

// Concrete Implementation 1: Razorpay Processor
type RazorpayProcessor struct {
    apiKey    string
    apiSecret string
    baseURL   string
}

func NewRazorpayProcessor(apiKey, apiSecret string) *RazorpayProcessor {
    return &RazorpayProcessor{
        apiKey:    apiKey,
        apiSecret: apiSecret,
        baseURL:   "https://api.razorpay.com/v1",
    }
}

func (r *RazorpayProcessor) ProcessTransaction(req *TransactionRequest) (*TransactionResult, error) {
    log.Printf("Razorpay: Processing %s payment for %s %s",
        req.PaymentType, req.Amount.String(), req.Currency)

    // Simulate Razorpay API call
    transactionID := fmt.Sprintf("rzp_%d", time.Now().UnixNano())

    // Calculate Razorpay fee (2% + ₹2)
    fee := req.Amount.Mul(decimal.NewFromFloat(0.02)).Add(decimal.NewFromInt(2))

    // Simulate processing delay
    time.Sleep(time.Millisecond * 100)

    return &TransactionResult{
        ID:          transactionID,
        Status:      "completed",
        Amount:      req.Amount,
        Fee:         fee,
        ProcessedAt: time.Now(),
        Reference:   fmt.Sprintf("RZP_REF_%d", time.Now().Unix()),
    }, nil
}

func (r *RazorpayProcessor) ValidateTransaction(req *ValidationRequest) error {
    switch req.PaymentType {
    case "credit_card":
        cardNumber, ok := req.PaymentData["card_number"].(string)
        if !ok || len(cardNumber) < 13 {
            return fmt.Errorf("invalid card number")
        }

        expiryDate, ok := req.PaymentData["expiry_date"].(string)
        if !ok || len(expiryDate) != 5 {
            return fmt.Errorf("invalid expiry date format")
        }

    case "digital_wallet":
        walletID, ok := req.PaymentData["wallet_id"].(string)
        if !ok || walletID == "" {
            return fmt.Errorf("invalid wallet ID")
        }

    default:
        return fmt.Errorf("unsupported payment type: %s", req.PaymentType)
    }

    return nil
}

func (r *RazorpayProcessor) GetProcessorInfo() *ProcessorInfo {
    return &ProcessorInfo{
        Name:         "Razorpay",
        Version:      "1.0.0",
        Capabilities: []string{"credit_card", "digital_wallet", "upi", "netbanking"},
        SupportedCurrencies: []string{"INR", "USD", "EUR"},
    }
}

// Concrete Implementation 2: Stripe Processor
type StripeProcessor struct {
    secretKey string
    baseURL   string
}

func NewStripeProcessor(secretKey string) *StripeProcessor {
    return &StripeProcessor{
        secretKey: secretKey,
        baseURL:   "https://api.stripe.com/v1",
    }
}

func (s *StripeProcessor) ProcessTransaction(req *TransactionRequest) (*TransactionResult, error) {
    log.Printf("Stripe: Processing %s payment for %s %s",
        req.PaymentType, req.Amount.String(), req.Currency)

    // Simulate Stripe API call
    transactionID := fmt.Sprintf("pi_%d", time.Now().UnixNano())

    // Calculate Stripe fee (2.9% + $0.30)
    fee := req.Amount.Mul(decimal.NewFromFloat(0.029)).Add(decimal.NewFromFloat(0.30))

    // Simulate processing delay
    time.Sleep(time.Millisecond * 150)

    return &TransactionResult{
        ID:          transactionID,
        Status:      "succeeded",
        Amount:      req.Amount,
        Fee:         fee,
        ProcessedAt: time.Now(),
        Reference:   fmt.Sprintf("STRIPE_REF_%d", time.Now().Unix()),
    }, nil
}

func (s *StripeProcessor) ValidateTransaction(req *ValidationRequest) error {
    switch req.PaymentType {
    case "credit_card":
        cardNumber, ok := req.PaymentData["card_number"].(string)
        if !ok || len(cardNumber) < 13 {
            return fmt.Errorf("invalid card number")
        }

        cvv, ok := req.PaymentData["cvv"].(string)
        if !ok || len(cvv) < 3 {
            return fmt.Errorf("invalid CVV")
        }

    case "digital_wallet":
        // Stripe has limited digital wallet support
        walletType, ok := req.PaymentData["wallet_type"].(string)
        if !ok || (walletType != "apple_pay" && walletType != "google_pay") {
            return fmt.Errorf("unsupported wallet type for Stripe: %s", walletType)
        }

    default:
        return fmt.Errorf("unsupported payment type: %s", req.PaymentType)
    }

    return nil
}

func (s *StripeProcessor) GetProcessorInfo() *ProcessorInfo {
    return &ProcessorInfo{
        Name:         "Stripe",
        Version:      "2023-10-16",
        Capabilities: []string{"credit_card", "apple_pay", "google_pay"},
        SupportedCurrencies: []string{"USD", "EUR", "GBP", "CAD", "AUD"},
    }
}

// Concrete Implementation 3: PayPal Processor
type PayPalProcessor struct {
    clientID     string
    clientSecret string
    baseURL      string
}

func NewPayPalProcessor(clientID, clientSecret string) *PayPalProcessor {
    return &PayPalProcessor{
        clientID:     clientID,
        clientSecret: clientSecret,
        baseURL:      "https://api.paypal.com/v1",
    }
}

func (p *PayPalProcessor) ProcessTransaction(req *TransactionRequest) (*TransactionResult, error) {
    log.Printf("PayPal: Processing %s payment for %s %s",
        req.PaymentType, req.Amount.String(), req.Currency)

    // Simulate PayPal API call
    transactionID := fmt.Sprintf("PAYID-%d", time.Now().UnixNano())

    // Calculate PayPal fee (variable rate, simplified to 3.4% + $0.30)
    fee := req.Amount.Mul(decimal.NewFromFloat(0.034)).Add(decimal.NewFromFloat(0.30))

    // Simulate processing delay
    time.Sleep(time.Millisecond * 200)

    return &TransactionResult{
        ID:          transactionID,
        Status:      "COMPLETED",
        Amount:      req.Amount,
        Fee:         fee,
        ProcessedAt: time.Now(),
        Reference:   fmt.Sprintf("PP_REF_%d", time.Now().Unix()),
    }, nil
}

func (p *PayPalProcessor) ValidateTransaction(req *ValidationRequest) error {
    switch req.PaymentType {
    case "credit_card":
        // PayPal can process credit cards
        cardNumber, ok := req.PaymentData["card_number"].(string)
        if !ok || len(cardNumber) < 13 {
            return fmt.Errorf("invalid card number")
        }

    case "digital_wallet":
        // PayPal itself is a digital wallet
        walletType, ok := req.PaymentData["wallet_type"].(string)
        if !ok || walletType != "paypal" {
            return fmt.Errorf("PayPal only supports PayPal wallet")
        }

    default:
        return fmt.Errorf("unsupported payment type: %s", req.PaymentType)
    }

    return nil
}

func (p *PayPalProcessor) GetProcessorInfo() *ProcessorInfo {
    return &ProcessorInfo{
        Name:         "PayPal",
        Version:      "1.0",
        Capabilities: []string{"credit_card", "paypal_wallet", "bank_transfer"},
        SupportedCurrencies: []string{"USD", "EUR", "GBP", "CAD", "AUD", "JPY"},
    }
}

// Payment Service that uses the Bridge pattern
type PaymentService struct {
    processors map[string]PaymentProcessor
}

func NewPaymentService() *PaymentService {
    return &PaymentService{
        processors: make(map[string]PaymentProcessor),
    }
}

func (ps *PaymentService) RegisterProcessor(name string, processor PaymentProcessor) {
    ps.processors[name] = processor
}

func (ps *PaymentService) CreatePaymentMethod(paymentType, processor string, config map[string]interface{}) (PaymentMethod, error) {
    proc, exists := ps.processors[processor]
    if !exists {
        return nil, fmt.Errorf("processor not found: %s", processor)
    }

    var paymentMethod PaymentMethod

    switch paymentType {
    case "credit_card":
        cardNumber, _ := config["card_number"].(string)
        expiryDate, _ := config["expiry_date"].(string)
        cvv, _ := config["cvv"].(string)
        holderName, _ := config["holder_name"].(string)

        paymentMethod = NewCreditCardPayment(cardNumber, expiryDate, cvv, holderName)

    case "digital_wallet":
        walletID, _ := config["wallet_id"].(string)
        walletType, _ := config["wallet_type"].(string)
        phoneNumber, _ := config["phone_number"].(string)

        paymentMethod = NewDigitalWalletPayment(walletID, walletType, phoneNumber)

    default:
        return nil, fmt.Errorf("unsupported payment type: %s", paymentType)
    }

    paymentMethod.SetProcessor(proc)
    return paymentMethod, nil
}

func (ps *PaymentService) GetAvailableProcessors() []string {
    var processors []string
    for name := range ps.processors {
        processors = append(processors, name)
    }
    return processors
}

// Example usage
func main() {
    fmt.Println("=== Bridge Pattern Demo ===\n")

    // Create payment service
    paymentService := NewPaymentService()

    // Register different processors
    paymentService.RegisterProcessor("razorpay", NewRazorpayProcessor("rzp_key", "rzp_secret"))
    paymentService.RegisterProcessor("stripe", NewStripeProcessor("sk_stripe_key"))
    paymentService.RegisterProcessor("paypal", NewPayPalProcessor("pp_client_id", "pp_secret"))

    // Example 1: Credit Card Payment with different processors
    fmt.Println("=== Credit Card Payments ===")

    creditCardConfig := map[string]interface{}{
        "card_number":  "4111111111111111",
        "expiry_date":  "12/25",
        "cvv":          "123",
        "holder_name":  "John Doe",
    }

    processors := []string{"razorpay", "stripe", "paypal"}
    amount := decimal.NewFromFloat(99.99)
    currency := "USD"

    for _, processor := range processors {
        fmt.Printf("\n--- Using %s processor ---\n", processor)

        paymentMethod, err := paymentService.CreatePaymentMethod("credit_card", processor, creditCardConfig)
        if err != nil {
            fmt.Printf("Error creating payment method: %v\n", err)
            continue
        }

        // Get payment details
        details := paymentMethod.GetPaymentDetails()
        fmt.Printf("Payment Details: %+v\n", details)

        // Process payment
        result, err := paymentMethod.ProcessPayment(amount, currency)
        if err != nil {
            fmt.Printf("Payment failed: %v\n", err)
            continue
        }

        fmt.Printf("Payment Result: %+v\n", result)
    }

    // Example 2: Digital Wallet Payment
    fmt.Println("\n\n=== Digital Wallet Payments ===")

    walletConfigs := []map[string]interface{}{
        {
            "wallet_id":    "user@paytm",
            "wallet_type":  "paytm",
            "phone_number": "+91-9876543210",
        },
        {
            "wallet_id":    "user@apple.com",
            "wallet_type":  "apple_pay",
            "phone_number": "+1-555-0123",
        },
        {
            "wallet_id":    "user@paypal.com",
            "wallet_type":  "paypal",
            "phone_number": "+1-555-0456",
        },
    }

    processorMap := map[string]string{
        "paytm":     "razorpay",
        "apple_pay": "stripe",
        "paypal":    "paypal",
    }

    for _, config := range walletConfigs {
        walletType := config["wallet_type"].(string)
        processor := processorMap[walletType]

        fmt.Printf("\n--- %s wallet with %s processor ---\n", walletType, processor)

        paymentMethod, err := paymentService.CreatePaymentMethod("digital_wallet", processor, config)
        if err != nil {
            fmt.Printf("Error creating payment method: %v\n", err)
            continue
        }

        // Process payment
        result, err := paymentMethod.ProcessPayment(decimal.NewFromFloat(50.00), "USD")
        if err != nil {
            fmt.Printf("Payment failed: %v\n", err)
            continue
        }

        fmt.Printf("Payment Result: %+v\n", result)
    }

    // Example 3: Runtime processor switching
    fmt.Println("\n\n=== Runtime Processor Switching ===")

    creditCard := NewCreditCardPayment("4111111111111111", "12/25", "123", "Jane Doe")

    for _, processor := range processors {
        fmt.Printf("\n--- Switching to %s ---\n", processor)

        proc := paymentService.processors[processor]
        creditCard.SetProcessor(proc)

        result, err := creditCard.ProcessPayment(decimal.NewFromFloat(25.00), "USD")
        if err != nil {
            fmt.Printf("Payment failed: %v\n", err)
            continue
        }

        fmt.Printf("Processor: %s, Transaction ID: %s, Fee: %s\n",
            proc.GetProcessorInfo().Name, result.TransactionID, result.Fee.String())
    }

    fmt.Println("\n=== Bridge Pattern Demo Complete ===")
}
```

## Variants & Trade-offs

### Variants

1. **Simple Bridge**

```go
type Shape interface {
    Draw()
    SetDrawingAPI(api DrawingAPI)
}

type DrawingAPI interface {
    DrawCircle(x, y, radius float64)
    DrawLine(x1, y1, x2, y2 float64)
}

type Circle struct {
    x, y, radius float64
    drawingAPI   DrawingAPI
}

func (c *Circle) Draw() {
    c.drawingAPI.DrawCircle(c.x, c.y, c.radius)
}
```

2. **Extended Bridge with Factory**

```go
type BridgeFactory struct {
    implementations map[string]Implementation
}

func (bf *BridgeFactory) CreateAbstraction(implType string) (Abstraction, error) {
    impl, exists := bf.implementations[implType]
    if !exists {
        return nil, fmt.Errorf("implementation not found: %s", implType)
    }

    abstraction := &ConcreteAbstraction{}
    abstraction.SetImplementation(impl)
    return abstraction, nil
}
```

3. **Bridge with Configuration**

```go
type ConfigurableBridge struct {
    abstraction    Abstraction
    implementation Implementation
    config         BridgeConfig
}

type BridgeConfig struct {
    RetryAttempts int
    Timeout       time.Duration
    EnableLogging bool
}

func (cb *ConfigurableBridge) Execute() error {
    for i := 0; i < cb.config.RetryAttempts; i++ {
        if cb.config.EnableLogging {
            log.Printf("Attempt %d", i+1)
        }

        if err := cb.abstraction.Operation(); err == nil {
            return nil
        }

        time.Sleep(cb.config.Timeout)
    }
    return fmt.Errorf("all attempts failed")
}
```

4. **Bridge with Middleware**

```go
type MiddlewareBridge struct {
    implementation Implementation
    middlewares    []Middleware
}

type Middleware interface {
    Before(req Request) error
    After(req Request, resp Response) error
}

func (mb *MiddlewareBridge) Execute(req Request) (Response, error) {
    // Execute before middlewares
    for _, middleware := range mb.middlewares {
        if err := middleware.Before(req); err != nil {
            return nil, err
        }
    }

    // Execute implementation
    resp, err := mb.implementation.Execute(req)

    // Execute after middlewares
    for i := len(mb.middlewares) - 1; i >= 0; i-- {
        mb.middlewares[i].After(req, resp)
    }

    return resp, err
}
```

### Trade-offs

**Pros:**

- **Decoupling**: Separates interface from implementation
- **Flexibility**: Both sides can vary independently
- **Runtime Switching**: Can change implementation at runtime
- **Platform Independence**: Hides platform-specific details
- **Extensibility**: Easy to add new abstractions and implementations

**Cons:**

- **Complexity**: Adds extra layer of indirection
- **Performance**: May introduce overhead
- **Learning Curve**: More complex than direct implementation
- **Debugging**: Harder to trace execution flow
- **Memory Usage**: Additional objects and references

**When to Choose Bridge vs Alternatives:**

| Scenario                 | Pattern               | Reason                       |
| ------------------------ | --------------------- | ---------------------------- |
| Multiple implementations | Bridge                | Independent variation        |
| Single implementation    | Direct Implementation | Simpler                      |
| Compile-time binding     | Template Method       | Better performance           |
| Runtime selection        | Strategy              | Similar but different intent |
| Hide complexity          | Facade                | Simplify interface           |

## Testable Example

```go
package main

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/stretchr/testify/mock"
    "github.com/shopspring/decimal"
)

// Mock Payment Processor for testing
type MockPaymentProcessor struct {
    mock.Mock
}

func (m *MockPaymentProcessor) ProcessTransaction(req *TransactionRequest) (*TransactionResult, error) {
    args := m.Called(req)
    return args.Get(0).(*TransactionResult), args.Error(1)
}

func (m *MockPaymentProcessor) ValidateTransaction(req *ValidationRequest) error {
    args := m.Called(req)
    return args.Error(0)
}

func (m *MockPaymentProcessor) GetProcessorInfo() *ProcessorInfo {
    args := m.Called()
    return args.Get(0).(*ProcessorInfo)
}

func TestCreditCardPayment_ProcessPayment(t *testing.T) {
    // Setup
    mockProcessor := &MockPaymentProcessor{}
    creditCard := NewCreditCardPayment("4111111111111111", "12/25", "123", "John Doe")
    creditCard.SetProcessor(mockProcessor)

    amount := decimal.NewFromFloat(100.0)
    currency := "USD"

    // Setup expectations
    expectedResult := &TransactionResult{
        ID:          "test_transaction_123",
        Status:      "completed",
        Amount:      amount,
        Fee:         decimal.NewFromFloat(2.3),
        ProcessedAt: time.Now(),
        Reference:   "TEST_REF_123",
    }

    mockProcessor.On("ValidateTransaction", mock.AnythingOfType("*main.ValidationRequest")).Return(nil)
    mockProcessor.On("ProcessTransaction", mock.AnythingOfType("*main.TransactionRequest")).Return(expectedResult, nil)
    mockProcessor.On("GetProcessorInfo").Return(&ProcessorInfo{Name: "MockProcessor"})

    // Execute
    result, err := creditCard.ProcessPayment(amount, currency)

    // Assert
    require.NoError(t, err)
    assert.Equal(t, expectedResult.ID, result.TransactionID)
    assert.Equal(t, expectedResult.Status, result.Status)
    assert.Equal(t, amount, result.Amount)
    assert.Equal(t, currency, result.Currency)
    assert.Equal(t, expectedResult.Fee, result.Fee)

    mockProcessor.AssertExpectations(t)
}

func TestCreditCardPayment_ValidatePayment(t *testing.T) {
    mockProcessor := &MockPaymentProcessor{}
    creditCard := NewCreditCardPayment("4111111111111111", "12/25", "123", "John Doe")
    creditCard.SetProcessor(mockProcessor)

    // Test successful validation
    mockProcessor.On("ValidateTransaction", mock.AnythingOfType("*main.ValidationRequest")).Return(nil)

    err := creditCard.ValidatePayment()
    assert.NoError(t, err)

    // Test validation failure
    mockProcessor.On("ValidateTransaction", mock.AnythingOfType("*main.ValidationRequest")).Return(fmt.Errorf("validation failed"))

    err = creditCard.ValidatePayment()
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "validation failed")

    mockProcessor.AssertExpectations(t)
}

func TestCreditCardPayment_NoProcessor(t *testing.T) {
    creditCard := NewCreditCardPayment("4111111111111111", "12/25", "123", "John Doe")

    // Test processing without processor
    _, err := creditCard.ProcessPayment(decimal.NewFromFloat(100.0), "USD")
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "no payment processor set")

    // Test validation without processor
    err = creditCard.ValidatePayment()
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "no payment processor set")
}

func TestDigitalWalletPayment_ProcessPayment(t *testing.T) {
    // Setup
    mockProcessor := &MockPaymentProcessor{}
    wallet := NewDigitalWalletPayment("user@wallet.com", "digital_wallet", "+1-555-0123")
    wallet.SetProcessor(mockProcessor)

    amount := decimal.NewFromFloat(50.0)
    currency := "USD"

    // Setup expectations
    expectedResult := &TransactionResult{
        ID:          "wallet_transaction_456",
        Status:      "completed",
        Amount:      amount,
        Fee:         decimal.NewFromFloat(1.5),
        ProcessedAt: time.Now(),
        Reference:   "WALLET_REF_456",
    }

    mockProcessor.On("ValidateTransaction", mock.AnythingOfType("*main.ValidationRequest")).Return(nil)
    mockProcessor.On("ProcessTransaction", mock.AnythingOfType("*main.TransactionRequest")).Return(expectedResult, nil)
    mockProcessor.On("GetProcessorInfo").Return(&ProcessorInfo{Name: "MockProcessor"})

    // Execute
    result, err := wallet.ProcessPayment(amount, currency)

    // Assert
    require.NoError(t, err)
    assert.Equal(t, expectedResult.ID, result.TransactionID)
    assert.Equal(t, expectedResult.Status, result.Status)
    assert.Equal(t, amount, result.Amount)
    assert.Equal(t, currency, result.Currency)

    mockProcessor.AssertExpectations(t)
}

func TestPaymentService_CreatePaymentMethod(t *testing.T) {
    paymentService := NewPaymentService()
    mockProcessor := &MockPaymentProcessor{}
    paymentService.RegisterProcessor("mock_processor", mockProcessor)

    t.Run("Create Credit Card Payment", func(t *testing.T) {
        config := map[string]interface{}{
            "card_number":  "4111111111111111",
            "expiry_date":  "12/25",
            "cvv":          "123",
            "holder_name":  "John Doe",
        }

        paymentMethod, err := paymentService.CreatePaymentMethod("credit_card", "mock_processor", config)

        assert.NoError(t, err)
        assert.NotNil(t, paymentMethod)

        details := paymentMethod.GetPaymentDetails()
        assert.Equal(t, "credit_card", details.PaymentType)
        assert.Contains(t, details.Details["masked_card_number"], "****")
    })

    t.Run("Create Digital Wallet Payment", func(t *testing.T) {
        config := map[string]interface{}{
            "wallet_id":    "user@wallet.com",
            "wallet_type":  "digital_wallet",
            "phone_number": "+1-555-0123",
        }

        paymentMethod, err := paymentService.CreatePaymentMethod("digital_wallet", "mock_processor", config)

        assert.NoError(t, err)
        assert.NotNil(t, paymentMethod)

        details := paymentMethod.GetPaymentDetails()
        assert.Equal(t, "digital_wallet", details.PaymentType)
        assert.Equal(t, "user@wallet.com", details.Details["wallet_id"])
    })

    t.Run("Unknown Processor", func(t *testing.T) {
        config := map[string]interface{}{}

        _, err := paymentService.CreatePaymentMethod("credit_card", "unknown_processor", config)

        assert.Error(t, err)
        assert.Contains(t, err.Error(), "processor not found")
    })

    t.Run("Unknown Payment Type", func(t *testing.T) {
        config := map[string]interface{}{}

        _, err := paymentService.CreatePaymentMethod("unknown_type", "mock_processor", config)

        assert.Error(t, err)
        assert.Contains(t, err.Error(), "unsupported payment type")
    })
}

func TestCreditCardPayment_CardTypeDetection(t *testing.T) {
    tests := []struct {
        cardNumber   string
        expectedType string
    }{
        {"4111111111111111", "visa"},
        {"5555555555554444", "mastercard"},
        {"378282246310005", "amex"},
        {"6011111111111117", "unknown"},
        {"123", "unknown"},
    }

    for _, tt := range tests {
        t.Run(tt.cardNumber, func(t *testing.T) {
            creditCard := NewCreditCardPayment(tt.cardNumber, "12/25", "123", "Test User")
            details := creditCard.GetPaymentDetails()

            assert.Equal(t, tt.expectedType, details.Details["card_type"])
        })
    }
}

func TestCreditCardPayment_CardMasking(t *testing.T) {
    creditCard := NewCreditCardPayment("4111111111111111", "12/25", "123", "Test User")
    details := creditCard.GetPaymentDetails()

    maskedCard := details.Details["masked_card_number"].(string)
    assert.Equal(t, "****-****-****-1111", maskedCard)

    // Test short card number
    shortCard := NewCreditCardPayment("123", "12/25", "123", "Test User")
    shortDetails := shortCard.GetPaymentDetails()
    shortMasked := shortDetails.Details["masked_card_number"].(string)
    assert.Equal(t, "****", shortMasked)
}

func TestRazorpayProcessor_ProcessTransaction(t *testing.T) {
    processor := NewRazorpayProcessor("test_key", "test_secret")

    req := &TransactionRequest{
        PaymentType: "credit_card",
        Amount:      decimal.NewFromFloat(100.0),
        Currency:    "INR",
        PaymentData: map[string]interface{}{
            "card_number": "4111111111111111",
        },
    }

    result, err := processor.ProcessTransaction(req)

    assert.NoError(t, err)
    assert.NotNil(t, result)
    assert.Contains(t, result.ID, "rzp_")
    assert.Equal(t, "completed", result.Status)
    assert.Equal(t, req.Amount, result.Amount)
    assert.True(t, result.Fee.GreaterThan(decimal.Zero))
}

func TestStripeProcessor_ProcessTransaction(t *testing.T) {
    processor := NewStripeProcessor("test_key")

    req := &TransactionRequest{
        PaymentType: "credit_card",
        Amount:      decimal.NewFromFloat(100.0),
        Currency:    "USD",
        PaymentData: map[string]interface{}{
            "card_number": "4111111111111111",
        },
    }

    result, err := processor.ProcessTransaction(req)

    assert.NoError(t, err)
    assert.NotNil(t, result)
    assert.Contains(t, result.ID, "pi_")
    assert.Equal(t, "succeeded", result.Status)
    assert.Equal(t, req.Amount, result.Amount)
    assert.True(t, result.Fee.GreaterThan(decimal.Zero))
}

func TestProcessorValidation(t *testing.T) {
    processors := map[string]PaymentProcessor{
        "razorpay": NewRazorpayProcessor("test_key", "test_secret"),
        "stripe":   NewStripeProcessor("test_key"),
        "paypal":   NewPayPalProcessor("client_id", "client_secret"),
    }

    validReq := &ValidationRequest{
        PaymentType: "credit_card",
        PaymentData: map[string]interface{}{
            "card_number":  "4111111111111111",
            "expiry_date":  "12/25",
            "cvv":          "123",
        },
    }

    invalidReq := &ValidationRequest{
        PaymentType: "credit_card",
        PaymentData: map[string]interface{}{
            "card_number": "123", // Invalid card number
        },
    }

    for name, processor := range processors {
        t.Run(name+"_valid", func(t *testing.T) {
            err := processor.ValidateTransaction(validReq)
            assert.NoError(t, err)
        })

        t.Run(name+"_invalid", func(t *testing.T) {
            err := processor.ValidateTransaction(invalidReq)
            assert.Error(t, err)
        })
    }
}

func BenchmarkCreditCardPayment_ProcessPayment(b *testing.B) {
    mockProcessor := &MockPaymentProcessor{}
    creditCard := NewCreditCardPayment("4111111111111111", "12/25", "123", "John Doe")
    creditCard.SetProcessor(mockProcessor)

    expectedResult := &TransactionResult{
        ID:          "benchmark_transaction",
        Status:      "completed",
        Amount:      decimal.NewFromFloat(100.0),
        Fee:         decimal.NewFromFloat(2.3),
        ProcessedAt: time.Now(),
    }

    mockProcessor.On("ValidateTransaction", mock.Anything).Return(nil)
    mockProcessor.On("ProcessTransaction", mock.Anything).Return(expectedResult, nil)
    mockProcessor.On("GetProcessorInfo").Return(&ProcessorInfo{Name: "MockProcessor"})

    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        _, err := creditCard.ProcessPayment(decimal.NewFromFloat(100.0), "USD")
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

## Integration Tips

### 1. Configuration Management

```go
type BridgeConfig struct {
    DefaultProcessor string                    `yaml:"default_processor"`
    Processors       map[string]ProcessorConfig `yaml:"processors"`
    PaymentMethods   []PaymentMethodConfig     `yaml:"payment_methods"`
}

type ProcessorConfig struct {
    Type   string                 `yaml:"type"`
    Config map[string]interface{} `yaml:"config"`
}

func LoadBridgeFromConfig(config BridgeConfig) (*PaymentService, error) {
    service := NewPaymentService()

    for name, procConfig := range config.Processors {
        processor, err := createProcessor(procConfig.Type, procConfig.Config)
        if err != nil {
            return nil, err
        }
        service.RegisterProcessor(name, processor)
    }

    return service, nil
}
```

### 2. Health Check Integration

```go
type HealthCheckableProcessor interface {
    PaymentProcessor
    HealthCheck() error
}

type HealthCheckBridge struct {
    abstraction Abstraction
    implementation HealthCheckableProcessor
}

func (h *HealthCheckBridge) CheckHealth() error {
    return h.implementation.HealthCheck()
}
```

### 3. Metrics Collection

```go
type MetricsBridge struct {
    abstraction Abstraction
    implementation Implementation
    metrics MetricsCollector
}

func (m *MetricsBridge) Execute() error {
    start := time.Now()
    err := m.abstraction.Execute()
    duration := time.Since(start)

    if err != nil {
        m.metrics.IncrementCounter("errors")
    } else {
        m.metrics.IncrementCounter("success")
    }

    m.metrics.RecordDuration("execution_time", duration)
    return err
}
```

### 4. Circuit Breaker Integration

```go
type CircuitBreakerBridge struct {
    abstraction    Abstraction
    implementation Implementation
    circuitBreaker *CircuitBreaker
}

func (c *CircuitBreakerBridge) Execute() error {
    return c.circuitBreaker.Execute(func() error {
        return c.abstraction.Execute()
    })
}
```

## Common Interview Questions

### 1. **How does Bridge differ from Adapter pattern?**

**Answer:**
| Aspect | Bridge | Adapter |
|--------|---------|---------|
| **Intent** | Separate abstraction from implementation | Make incompatible interfaces work together |
| **Design Time** | Designed upfront for flexibility | Added when interface mismatch is discovered |
| **Structure** | Both hierarchies can evolve independently | Typically adapts existing interface |
| **Use Case** | Platform independence, multiple implementations | Legacy integration, third-party libraries |

**Example:**

```go
// Bridge - designed for multiple implementations
type PaymentMethod interface {
    Process(amount decimal.Decimal) error
    SetProcessor(proc PaymentProcessor) // Bridge connection
}

type PaymentProcessor interface {
    Execute(req ProcessRequest) error
}

// Adapter - adapts existing interface
type LegacyPaymentAdapter struct {
    legacyGateway *LegacyGateway
}

func (l *LegacyPaymentAdapter) Process(req ModernRequest) error {
    legacyReq := l.convertToLegacy(req)
    return l.legacyGateway.ProcessOldStyle(legacyReq)
}
```

### 2. **When would you choose Bridge over Strategy pattern?**

**Answer:**
**Bridge**: Use when you have **abstraction hierarchy** and **implementation hierarchy** that need to vary independently.

**Strategy**: Use when you have **different algorithms** for the same task that can be swapped at runtime.

```go
// Bridge - abstraction and implementation hierarchies
type Shape interface { // Abstraction hierarchy
    Draw()
    SetRenderer(r Renderer)
}

type Circle struct { // Refined abstraction
    renderer Renderer
}

type Renderer interface { // Implementation hierarchy
    RenderCircle()
}

type OpenGLRenderer struct{} // Concrete implementation

// Strategy - different algorithms for same task
type SortContext struct {
    strategy SortStrategy
}

type SortStrategy interface {
    Sort(data []int) []int
}

type QuickSort struct{}
type MergeSort struct{}
```

**Bridge**: "I have different shapes that can be rendered on different platforms"
**Strategy**: "I have different sorting algorithms I can choose from"

### 3. **How do you handle multiple implementations in Bridge pattern?**

**Answer:**
Use factory patterns and registries:

1. **Implementation Registry**:

```go
type ImplementationRegistry struct {
    implementations map[string]Implementation
}

func (r *ImplementationRegistry) Register(name string, impl Implementation) {
    r.implementations[name] = impl
}

func (r *ImplementationRegistry) Get(name string) Implementation {
    return r.implementations[name]
}
```

2. **Factory with Bridge**:

```go
type BridgeFactory struct {
    registry *ImplementationRegistry
}

func (bf *BridgeFactory) CreateAbstraction(abstractionType, implType string) Abstraction {
    impl := bf.registry.Get(implType)

    switch abstractionType {
    case "payment":
        payment := &PaymentAbstraction{}
        payment.SetImplementation(impl)
        return payment
    }

    return nil
}
```

3. **Configuration-Driven Selection**:

```go
type BridgeSelector struct {
    config map[string]string // abstraction -> implementation mapping
    registry *ImplementationRegistry
}

func (bs *BridgeSelector) CreateBridge(abstractionType string) Abstraction {
    implType := bs.config[abstractionType]
    impl := bs.registry.Get(implType)

    abstraction := createAbstraction(abstractionType)
    abstraction.SetImplementation(impl)
    return abstraction
}
```

### 4. **How do you test Bridge pattern implementations?**

**Answer:**
Focus on testing the bridge connection and both hierarchies:

1. **Test Abstraction with Mock Implementation**:

```go
func TestPaymentMethod_WithMockProcessor(t *testing.T) {
    mockProcessor := &MockPaymentProcessor{}
    payment := &CreditCardPayment{}
    payment.SetProcessor(mockProcessor)

    mockProcessor.On("Process", mock.Anything).Return(nil)

    err := payment.ProcessPayment(decimal.NewFromFloat(100))
    assert.NoError(t, err)
    mockProcessor.AssertExpectations(t)
}
```

2. **Test Implementation Independence**:

```go
func TestPaymentMethod_ProcessorIndependence(t *testing.T) {
    payment := &CreditCardPayment{}

    // Test with different processors
    processors := []PaymentProcessor{
        &RazorpayProcessor{},
        &StripeProcessor{},
        &PayPalProcessor{},
    }

    for _, proc := range processors {
        payment.SetProcessor(proc)
        err := payment.ProcessPayment(decimal.NewFromFloat(100))
        assert.NoError(t, err)
    }
}
```

3. **Test Bridge Flexibility**:

```go
func TestBridge_RuntimeSwitching(t *testing.T) {
    abstraction := &PaymentAbstraction{}

    // Start with one implementation
    impl1 := &ProcessorA{}
    abstraction.SetImplementation(impl1)
    result1 := abstraction.Execute()

    // Switch to another implementation
    impl2 := &ProcessorB{}
    abstraction.SetImplementation(impl2)
    result2 := abstraction.Execute()

    // Results should be different based on implementation
    assert.NotEqual(t, result1, result2)
}
```

### 5. **How do you handle errors across the bridge?**

**Answer:**
Implement consistent error handling strategies:

1. **Error Translation**:

```go
type BridgeErrorTranslator struct {
    abstraction Abstraction
    implementation Implementation
}

func (b *BridgeErrorTranslator) Execute() error {
    err := b.abstraction.Execute()
    if err != nil {
        // Translate implementation-specific errors to abstraction errors
        return b.translateError(err)
    }
    return nil
}

func (b *BridgeErrorTranslator) translateError(err error) error {
    switch {
    case strings.Contains(err.Error(), "network"):
        return NewAbstractionError("NETWORK_ERROR", err)
    case strings.Contains(err.Error(), "auth"):
        return NewAbstractionError("AUTH_ERROR", err)
    default:
        return NewAbstractionError("UNKNOWN_ERROR", err)
    }
}
```

2. **Error Wrapping**:

```go
type AbstractionError struct {
    Code           string
    Message        string
    ImplementationError error
    Context        map[string]interface{}
}

func (ae *AbstractionError) Error() string {
    return fmt.Sprintf("[%s] %s: %v", ae.Code, ae.Message, ae.ImplementationError)
}

func (ae *AbstractionError) Unwrap() error {
    return ae.ImplementationError
}
```

3. **Retry Logic Across Bridge**:

```go
type RetryableBridge struct {
    abstraction Abstraction
    maxRetries  int
    backoff     time.Duration
}

func (rb *RetryableBridge) Execute() error {
    var lastErr error

    for i := 0; i < rb.maxRetries; i++ {
        err := rb.abstraction.Execute()
        if err == nil {
            return nil
        }

        lastErr = err
        if !rb.isRetryable(err) {
            break
        }

        time.Sleep(rb.backoff * time.Duration(i+1))
    }

    return lastErr
}

func (rb *RetryableBridge) isRetryable(err error) bool {
    // Determine if error is retryable based on error type or message
    return strings.Contains(err.Error(), "timeout") ||
           strings.Contains(err.Error(), "network")
}
```
