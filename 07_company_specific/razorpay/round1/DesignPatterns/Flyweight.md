---
# Auto-generated front matter
Title: Flyweight
LastUpdated: 2025-11-06T20:45:58.529194
Tags: []
Status: draft
---

# Flyweight Pattern

## Pattern Name & Intent

**Flyweight** is a structural design pattern that lets you fit more objects into the available amount of RAM by sharing efficiently the common parts of state between multiple objects instead of keeping it all in each object.

**Key Intent:**

- Minimize memory usage when dealing with large numbers of similar objects
- Share common data between multiple objects efficiently
- Separate intrinsic state (shared) from extrinsic state (unique per object)
- Reduce object creation costs through reuse
- Support large numbers of fine-grained objects efficiently

## When to Use

**Use Flyweight when:**

1. **Large Object Collections**: Need to support large numbers of similar objects
2. **Memory Constraints**: Storage costs are high due to object quantity
3. **Shared State**: Objects have significant shared/intrinsic state
4. **Object Identity**: Object identity is not important for the application
5. **Extrinsic State**: Object-specific state can be computed or passed as parameters
6. **Performance Critical**: Object creation/destruction is expensive
7. **Caching Benefits**: Shared objects can be cached effectively

**Don't use when:**

- Object state is mostly unique (little intrinsic state to share)
- Memory usage is not a concern
- Object identity is important
- Complexity outweighs memory benefits
- Thread safety requirements are complex

## Real-World Use Cases (Payments/Fintech)

### 1. Currency and Exchange Rate Management

```go
// Flyweight for currency information
type CurrencyFlyweight interface {
    GetExchangeRate(baseCurrency string, targetCurrency string, date time.Time) (decimal.Decimal, error)
    FormatAmount(amount decimal.Decimal, locale string) string
    GetSymbol() string
    GetDecimalPlaces() int
}

// Intrinsic state - shared among all instances
type Currency struct {
    Code          string // USD, EUR, GBP, etc.
    Symbol        string // $, €, £, etc.
    DecimalPlaces int    // 2 for most currencies
    Name          string // US Dollar, Euro, etc.
}

func (c *Currency) GetSymbol() string {
    return c.Symbol
}

func (c *Currency) GetDecimalPlaces() int {
    return c.DecimalPlaces
}

func (c *Currency) FormatAmount(amount decimal.Decimal, locale string) string {
    // Format based on currency and locale
    switch c.Code {
    case "USD":
        return fmt.Sprintf("$%s", amount.StringFixed(2))
    case "EUR":
        return fmt.Sprintf("€%s", amount.StringFixed(2))
    case "GBP":
        return fmt.Sprintf("£%s", amount.StringFixed(2))
    default:
        return fmt.Sprintf("%s %s", c.Symbol, amount.StringFixed(int32(c.DecimalPlaces)))
    }
}

func (c *Currency) GetExchangeRate(baseCurrency string, targetCurrency string, date time.Time) (decimal.Decimal, error) {
    // This would typically fetch from an external service or cache
    // For demo purposes, return mock rates
    rates := map[string]map[string]decimal.Decimal{
        "USD": {
            "EUR": decimal.NewFromFloat(0.85),
            "GBP": decimal.NewFromFloat(0.73),
            "JPY": decimal.NewFromFloat(110.0),
        },
        "EUR": {
            "USD": decimal.NewFromFloat(1.18),
            "GBP": decimal.NewFromFloat(0.86),
        },
    }

    if baseRates, exists := rates[baseCurrency]; exists {
        if rate, exists := baseRates[targetCurrency]; exists {
            return rate, nil
        }
    }

    return decimal.Zero, fmt.Errorf("exchange rate not found for %s to %s", baseCurrency, targetCurrency)
}

// Flyweight factory
type CurrencyFactory struct {
    currencies map[string]*Currency
    mu         sync.RWMutex
}

func NewCurrencyFactory() *CurrencyFactory {
    return &CurrencyFactory{
        currencies: make(map[string]*Currency),
    }
}

func (f *CurrencyFactory) GetCurrency(code string) *Currency {
    f.mu.RLock()
    if currency, exists := f.currencies[code]; exists {
        f.mu.RUnlock()
        return currency
    }
    f.mu.RUnlock()

    f.mu.Lock()
    defer f.mu.Unlock()

    // Double-check after acquiring write lock
    if currency, exists := f.currencies[code]; exists {
        return currency
    }

    // Create new currency flyweight
    currency := f.createCurrency(code)
    f.currencies[code] = currency
    return currency
}

func (f *CurrencyFactory) createCurrency(code string) *Currency {
    currencyData := map[string]Currency{
        "USD": {Code: "USD", Symbol: "$", DecimalPlaces: 2, Name: "US Dollar"},
        "EUR": {Code: "EUR", Symbol: "€", DecimalPlaces: 2, Name: "Euro"},
        "GBP": {Code: "GBP", Symbol: "£", DecimalPlaces: 2, Name: "British Pound"},
        "JPY": {Code: "JPY", Symbol: "¥", DecimalPlaces: 0, Name: "Japanese Yen"},
        "BTC": {Code: "BTC", Symbol: "₿", DecimalPlaces: 8, Name: "Bitcoin"},
    }

    if data, exists := currencyData[code]; exists {
        return &data
    }

    // Default currency for unknown codes
    return &Currency{
        Code:          code,
        Symbol:        code,
        DecimalPlaces: 2,
        Name:          fmt.Sprintf("Unknown Currency (%s)", code),
    }
}

func (f *CurrencyFactory) GetCreatedCurrenciesCount() int {
    f.mu.RLock()
    defer f.mu.RUnlock()
    return len(f.currencies)
}

// Context - contains extrinsic state
type MoneyAmount struct {
    Amount   decimal.Decimal
    Currency *Currency // Flyweight reference
    Date     time.Time
}

func NewMoneyAmount(amount decimal.Decimal, currencyCode string, factory *CurrencyFactory) *MoneyAmount {
    return &MoneyAmount{
        Amount:   amount,
        Currency: factory.GetCurrency(currencyCode),
        Date:     time.Now(),
    }
}

func (m *MoneyAmount) Format(locale string) string {
    return m.Currency.FormatAmount(m.Amount, locale)
}

func (m *MoneyAmount) ConvertTo(targetCurrencyCode string, factory *CurrencyFactory) (*MoneyAmount, error) {
    targetCurrency := factory.GetCurrency(targetCurrencyCode)

    if m.Currency.Code == targetCurrency.Code {
        return m, nil // Same currency
    }

    rate, err := m.Currency.GetExchangeRate(m.Currency.Code, targetCurrency.Code, m.Date)
    if err != nil {
        return nil, err
    }

    convertedAmount := m.Amount.Mul(rate)

    return &MoneyAmount{
        Amount:   convertedAmount,
        Currency: targetCurrency,
        Date:     time.Now(),
    }, nil
}
```

### 2. Transaction Type Flyweights

```go
// Transaction type flyweight for processing rules
type TransactionTypeFlyweight interface {
    CalculateFee(amount decimal.Decimal, merchantCategory string) decimal.Decimal
    ValidateTransaction(transaction *Transaction) error
    GetProcessingTime() time.Duration
    RequiresApproval(amount decimal.Decimal) bool
}

// Intrinsic state - shared transaction processing rules
type TransactionType struct {
    TypeCode         string
    Name             string
    BaseFeePercent   decimal.Decimal
    FixedFee         decimal.Decimal
    MaxProcessingTime time.Duration
    ApprovalThreshold decimal.Decimal
    AllowedMerchantCategories []string
}

func (tt *TransactionType) CalculateFee(amount decimal.Decimal, merchantCategory string) decimal.Decimal {
    baseFee := amount.Mul(tt.BaseFeePercent.Div(decimal.NewFromInt(100)))
    totalFee := baseFee.Add(tt.FixedFee)

    // Apply merchant category modifiers
    modifier := tt.getMerchantCategoryModifier(merchantCategory)
    return totalFee.Mul(modifier)
}

func (tt *TransactionType) getMerchantCategoryModifier(category string) decimal.Decimal {
    modifiers := map[string]decimal.Decimal{
        "GROCERY":     decimal.NewFromFloat(0.8),  // 20% discount
        "GAS_STATION": decimal.NewFromFloat(0.9),  // 10% discount
        "RESTAURANT":  decimal.NewFromFloat(1.0),  // No change
        "LUXURY":      decimal.NewFromFloat(1.2),  // 20% surcharge
        "HIGH_RISK":   decimal.NewFromFloat(1.5),  // 50% surcharge
    }

    if modifier, exists := modifiers[category]; exists {
        return modifier
    }
    return decimal.NewFromFloat(1.0) // Default: no change
}

func (tt *TransactionType) ValidateTransaction(transaction *Transaction) error {
    // Check if merchant category is allowed
    if len(tt.AllowedMerchantCategories) > 0 {
        allowed := false
        for _, category := range tt.AllowedMerchantCategories {
            if category == transaction.MerchantCategory {
                allowed = true
                break
            }
        }
        if !allowed {
            return fmt.Errorf("merchant category %s not allowed for transaction type %s",
                transaction.MerchantCategory, tt.TypeCode)
        }
    }

    // Additional validation rules based on transaction type
    switch tt.TypeCode {
    case "CREDIT_CARD":
        return tt.validateCreditCard(transaction)
    case "DEBIT_CARD":
        return tt.validateDebitCard(transaction)
    case "BANK_TRANSFER":
        return tt.validateBankTransfer(transaction)
    }

    return nil
}

func (tt *TransactionType) validateCreditCard(transaction *Transaction) error {
    if transaction.Amount.GreaterThan(decimal.NewFromInt(10000)) {
        return fmt.Errorf("credit card transaction amount exceeds limit")
    }
    return nil
}

func (tt *TransactionType) validateDebitCard(transaction *Transaction) error {
    // Debit cards typically have lower limits
    if transaction.Amount.GreaterThan(decimal.NewFromInt(5000)) {
        return fmt.Errorf("debit card transaction amount exceeds limit")
    }
    return nil
}

func (tt *TransactionType) validateBankTransfer(transaction *Transaction) error {
    // Bank transfers may have different validation rules
    if transaction.Amount.LessThan(decimal.NewFromInt(1)) {
        return fmt.Errorf("bank transfer minimum amount not met")
    }
    return nil
}

func (tt *TransactionType) GetProcessingTime() time.Duration {
    return tt.MaxProcessingTime
}

func (tt *TransactionType) RequiresApproval(amount decimal.Decimal) bool {
    return amount.GreaterThan(tt.ApprovalThreshold)
}

// Transaction type factory
type TransactionTypeFactory struct {
    types map[string]*TransactionType
    mu    sync.RWMutex
}

func NewTransactionTypeFactory() *TransactionTypeFactory {
    factory := &TransactionTypeFactory{
        types: make(map[string]*TransactionType),
    }

    // Pre-populate common transaction types
    factory.initializeCommonTypes()
    return factory
}

func (f *TransactionTypeFactory) initializeCommonTypes() {
    commonTypes := map[string]*TransactionType{
        "CREDIT_CARD": {
            TypeCode:         "CREDIT_CARD",
            Name:             "Credit Card Payment",
            BaseFeePercent:   decimal.NewFromFloat(2.9),
            FixedFee:         decimal.NewFromFloat(0.30),
            MaxProcessingTime: 5 * time.Second,
            ApprovalThreshold: decimal.NewFromInt(10000),
            AllowedMerchantCategories: []string{"GROCERY", "RESTAURANT", "RETAIL", "GAS_STATION"},
        },
        "DEBIT_CARD": {
            TypeCode:         "DEBIT_CARD",
            Name:             "Debit Card Payment",
            BaseFeePercent:   decimal.NewFromFloat(1.9),
            FixedFee:         decimal.NewFromFloat(0.25),
            MaxProcessingTime: 3 * time.Second,
            ApprovalThreshold: decimal.NewFromInt(5000),
            AllowedMerchantCategories: []string{"GROCERY", "RESTAURANT", "RETAIL", "GAS_STATION"},
        },
        "BANK_TRANSFER": {
            TypeCode:         "BANK_TRANSFER",
            Name:             "Bank Transfer",
            BaseFeePercent:   decimal.NewFromFloat(0.5),
            FixedFee:         decimal.NewFromFloat(1.00),
            MaxProcessingTime: 24 * time.Hour,
            ApprovalThreshold: decimal.NewFromInt(50000),
            AllowedMerchantCategories: []string{}, // All categories allowed
        },
        "CRYPTOCURRENCY": {
            TypeCode:         "CRYPTOCURRENCY",
            Name:             "Cryptocurrency Payment",
            BaseFeePercent:   decimal.NewFromFloat(1.0),
            FixedFee:         decimal.NewFromFloat(0.00),
            MaxProcessingTime: 10 * time.Minute,
            ApprovalThreshold: decimal.NewFromInt(25000),
            AllowedMerchantCategories: []string{"DIGITAL_GOODS", "SOFTWARE", "GAMING"},
        },
    }

    for code, transactionType := range commonTypes {
        f.types[code] = transactionType
    }
}

func (f *TransactionTypeFactory) GetTransactionType(typeCode string) *TransactionType {
    f.mu.RLock()
    if transactionType, exists := f.types[typeCode]; exists {
        f.mu.RUnlock()
        return transactionType
    }
    f.mu.RUnlock()

    f.mu.Lock()
    defer f.mu.Unlock()

    // Double-check after acquiring write lock
    if transactionType, exists := f.types[typeCode]; exists {
        return transactionType
    }

    // Create default transaction type for unknown codes
    defaultType := &TransactionType{
        TypeCode:         typeCode,
        Name:             fmt.Sprintf("Unknown Transaction Type (%s)", typeCode),
        BaseFeePercent:   decimal.NewFromFloat(3.0), // Higher default fee
        FixedFee:         decimal.NewFromFloat(0.50),
        MaxProcessingTime: 30 * time.Second,
        ApprovalThreshold: decimal.NewFromInt(1000), // Lower approval threshold
        AllowedMerchantCategories: []string{},
    }

    f.types[typeCode] = defaultType
    return defaultType
}

func (f *TransactionTypeFactory) GetLoadedTypesCount() int {
    f.mu.RLock()
    defer f.mu.RUnlock()
    return len(f.types)
}

// Context with extrinsic state
type Transaction struct {
    ID               string
    Amount           decimal.Decimal
    Currency         *Currency
    Type             *TransactionType // Flyweight reference
    MerchantID       string
    MerchantCategory string
    CustomerID       string
    Timestamp        time.Time
    ExtrinsicData    map[string]interface{} // Context-specific data
}

func NewTransaction(
    amount decimal.Decimal,
    currencyCode string,
    typeCode string,
    merchantID string,
    merchantCategory string,
    customerID string,
    currencyFactory *CurrencyFactory,
    typeFactory *TransactionTypeFactory,
) *Transaction {
    return &Transaction{
        ID:               generateTransactionID(),
        Amount:           amount,
        Currency:         currencyFactory.GetCurrency(currencyCode),
        Type:             typeFactory.GetTransactionType(typeCode),
        MerchantID:       merchantID,
        MerchantCategory: merchantCategory,
        CustomerID:       customerID,
        Timestamp:        time.Now(),
        ExtrinsicData:    make(map[string]interface{}),
    }
}

func (t *Transaction) CalculateFee() decimal.Decimal {
    return t.Type.CalculateFee(t.Amount, t.MerchantCategory)
}

func (t *Transaction) Validate() error {
    return t.Type.ValidateTransaction(t)
}

func (t *Transaction) RequiresApproval() bool {
    return t.Type.RequiresApproval(t.Amount)
}

func (t *Transaction) GetProcessingTime() time.Duration {
    return t.Type.GetProcessingTime()
}
```

### 3. Risk Profile Flyweights

```go
// Risk profile flyweight for fraud detection
type RiskProfileFlyweight interface {
    CalculateRiskScore(context *RiskContext) float64
    GetRiskThresholds() *RiskThresholds
    RequiresAdditionalVerification(score float64) bool
    GetRiskFactors() []RiskFactor
}

type RiskFactor struct {
    Name        string
    Weight      float64
    MaxScore    float64
    Description string
}

type RiskThresholds struct {
    Low    float64
    Medium float64
    High   float64
}

// Intrinsic state - shared risk calculation logic
type RiskProfile struct {
    ProfileCode string
    Name        string
    Factors     []RiskFactor
    Thresholds  *RiskThresholds
    BaselineRisk float64
}

func (rp *RiskProfile) CalculateRiskScore(context *RiskContext) float64 {
    score := rp.BaselineRisk

    for _, factor := range rp.Factors {
        factorScore := rp.calculateFactorScore(factor, context)
        score += factorScore * factor.Weight
    }

    // Normalize score to 0-1 range
    if score > 1.0 {
        score = 1.0
    } else if score < 0.0 {
        score = 0.0
    }

    return score
}

func (rp *RiskProfile) calculateFactorScore(factor RiskFactor, context *RiskContext) float64 {
    switch factor.Name {
    case "TRANSACTION_AMOUNT":
        return rp.calculateAmountRisk(context.Amount, context.CustomerAvgTransaction)
    case "TRANSACTION_FREQUENCY":
        return rp.calculateFrequencyRisk(context.RecentTransactionCount, context.CustomerAvgFrequency)
    case "GEOGRAPHIC_LOCATION":
        return rp.calculateLocationRisk(context.Location, context.CustomerUsualLocations)
    case "TIME_OF_DAY":
        return rp.calculateTimeRisk(context.TransactionTime, context.CustomerUsualTimes)
    case "MERCHANT_RISK":
        return rp.calculateMerchantRisk(context.MerchantRiskScore)
    case "DEVICE_FINGERPRINT":
        return rp.calculateDeviceRisk(context.DeviceFingerprint, context.CustomerDevices)
    default:
        return 0.0
    }
}

func (rp *RiskProfile) calculateAmountRisk(amount, avgAmount decimal.Decimal) float64 {
    if avgAmount.IsZero() {
        return 0.5 // Neutral risk for new customers
    }

    ratio := amount.Div(avgAmount).InexactFloat64()

    switch {
    case ratio > 10.0:
        return 1.0 // Very high risk
    case ratio > 5.0:
        return 0.8
    case ratio > 2.0:
        return 0.4
    case ratio < 0.1:
        return 0.3 // Unusually small amounts can also be suspicious
    default:
        return 0.1 // Normal range
    }
}

func (rp *RiskProfile) calculateFrequencyRisk(recentCount int, avgFrequency float64) float64 {
    if avgFrequency == 0 {
        return 0.3
    }

    frequencyRatio := float64(recentCount) / avgFrequency

    switch {
    case frequencyRatio > 5.0:
        return 1.0
    case frequencyRatio > 3.0:
        return 0.7
    case frequencyRatio > 2.0:
        return 0.4
    default:
        return 0.1
    }
}

func (rp *RiskProfile) calculateLocationRisk(location string, usualLocations []string) float64 {
    for _, usual := range usualLocations {
        if usual == location {
            return 0.1 // Known location
        }
    }

    // Check if it's a neighboring location (simplified)
    for _, usual := range usualLocations {
        if rp.isNeighboringLocation(location, usual) {
            return 0.3
        }
    }

    return 0.8 // Completely new location
}

func (rp *RiskProfile) isNeighboringLocation(loc1, loc2 string) bool {
    // Simplified neighbor check - in reality, would use geolocation
    return strings.HasPrefix(loc1, loc2[:2]) || strings.HasPrefix(loc2, loc1[:2])
}

func (rp *RiskProfile) calculateTimeRisk(transactionTime time.Time, usualTimes []time.Time) float64 {
    hour := transactionTime.Hour()

    // Check if it's within usual hours
    for _, usual := range usualTimes {
        usualHour := usual.Hour()
        if abs(hour-usualHour) <= 2 { // Within 2 hours
            return 0.1
        }
    }

    // Check if it's during typical business hours
    if hour >= 9 && hour <= 17 {
        return 0.3
    }

    // Late night/early morning transactions are riskier
    if hour >= 22 || hour <= 6 {
        return 0.8
    }

    return 0.4
}

func (rp *RiskProfile) calculateMerchantRisk(merchantScore float64) float64 {
    return merchantScore
}

func (rp *RiskProfile) calculateDeviceRisk(deviceFingerprint string, customerDevices []string) float64 {
    for _, device := range customerDevices {
        if device == deviceFingerprint {
            return 0.1 // Known device
        }
    }
    return 0.7 // New device
}

func (rp *RiskProfile) GetRiskThresholds() *RiskThresholds {
    return rp.Thresholds
}

func (rp *RiskProfile) RequiresAdditionalVerification(score float64) bool {
    return score >= rp.Thresholds.Medium
}

func (rp *RiskProfile) GetRiskFactors() []RiskFactor {
    return rp.Factors
}

// Risk profile factory
type RiskProfileFactory struct {
    profiles map[string]*RiskProfile
    mu       sync.RWMutex
}

func NewRiskProfileFactory() *RiskProfileFactory {
    factory := &RiskProfileFactory{
        profiles: make(map[string]*RiskProfile),
    }

    factory.initializeStandardProfiles()
    return factory
}

func (f *RiskProfileFactory) initializeStandardProfiles() {
    profiles := map[string]*RiskProfile{
        "STANDARD": {
            ProfileCode:  "STANDARD",
            Name:         "Standard Risk Profile",
            BaselineRisk: 0.1,
            Thresholds: &RiskThresholds{
                Low:    0.3,
                Medium: 0.6,
                High:   0.8,
            },
            Factors: []RiskFactor{
                {Name: "TRANSACTION_AMOUNT", Weight: 0.3, MaxScore: 1.0, Description: "Transaction amount vs historical average"},
                {Name: "TRANSACTION_FREQUENCY", Weight: 0.2, MaxScore: 1.0, Description: "Recent transaction frequency"},
                {Name: "GEOGRAPHIC_LOCATION", Weight: 0.2, MaxScore: 1.0, Description: "Geographic location analysis"},
                {Name: "TIME_OF_DAY", Weight: 0.1, MaxScore: 1.0, Description: "Time of day analysis"},
                {Name: "MERCHANT_RISK", Weight: 0.15, MaxScore: 1.0, Description: "Merchant risk score"},
                {Name: "DEVICE_FINGERPRINT", Weight: 0.05, MaxScore: 1.0, Description: "Device fingerprint analysis"},
            },
        },
        "HIGH_VALUE": {
            ProfileCode:  "HIGH_VALUE",
            Name:         "High Value Transaction Profile",
            BaselineRisk: 0.2,
            Thresholds: &RiskThresholds{
                Low:    0.2,
                Medium: 0.5,
                High:   0.7,
            },
            Factors: []RiskFactor{
                {Name: "TRANSACTION_AMOUNT", Weight: 0.4, MaxScore: 1.0, Description: "High emphasis on amount"},
                {Name: "GEOGRAPHIC_LOCATION", Weight: 0.3, MaxScore: 1.0, Description: "Location is critical for high-value"},
                {Name: "DEVICE_FINGERPRINT", Weight: 0.2, MaxScore: 1.0, Description: "Device verification important"},
                {Name: "TIME_OF_DAY", Weight: 0.1, MaxScore: 1.0, Description: "Time analysis"},
            },
        },
        "INTERNATIONAL": {
            ProfileCode:  "INTERNATIONAL",
            Name:         "International Transaction Profile",
            BaselineRisk: 0.3,
            Thresholds: &RiskThresholds{
                Low:    0.4,
                Medium: 0.7,
                High:   0.9,
            },
            Factors: []RiskFactor{
                {Name: "GEOGRAPHIC_LOCATION", Weight: 0.5, MaxScore: 1.0, Description: "Geographic location is primary factor"},
                {Name: "TRANSACTION_AMOUNT", Weight: 0.2, MaxScore: 1.0, Description: "Amount analysis"},
                {Name: "MERCHANT_RISK", Weight: 0.2, MaxScore: 1.0, Description: "International merchant risk"},
                {Name: "DEVICE_FINGERPRINT", Weight: 0.1, MaxScore: 1.0, Description: "Device verification"},
            },
        },
    }

    for code, profile := range profiles {
        f.profiles[code] = profile
    }
}

func (f *RiskProfileFactory) GetRiskProfile(profileCode string) *RiskProfile {
    f.mu.RLock()
    if profile, exists := f.profiles[profileCode]; exists {
        f.mu.RUnlock()
        return profile
    }
    f.mu.RUnlock()

    f.mu.Lock()
    defer f.mu.Unlock()

    // Double-check after acquiring write lock
    if profile, exists := f.profiles[profileCode]; exists {
        return profile
    }

    // Return standard profile for unknown codes
    return f.profiles["STANDARD"]
}

// Extrinsic state - context for risk calculation
type RiskContext struct {
    Amount                    decimal.Decimal
    CustomerAvgTransaction    decimal.Decimal
    RecentTransactionCount    int
    CustomerAvgFrequency      float64
    Location                  string
    CustomerUsualLocations    []string
    TransactionTime           time.Time
    CustomerUsualTimes        []time.Time
    MerchantRiskScore         float64
    DeviceFingerprint         string
    CustomerDevices           []string
}

// Helper function
func abs(x int) int {
    if x < 0 {
        return -x
    }
    return x
}
```

## Go Implementation

```go
package main

import (
    "fmt"
    "sync"
    "time"
    "strings"
    "github.com/shopspring/decimal"
)

// Flyweight interface
type CharacterFlyweight interface {
    Render(x, y int, font string, size int, color string) string
}

// Concrete flyweight - intrinsic state
type Character struct {
    Symbol rune // The actual character - this is intrinsic state
}

func (c *Character) Render(x, y int, font string, size int, color string) string {
    return fmt.Sprintf("Rendering '%c' at (%d,%d) with font=%s, size=%d, color=%s",
        c.Symbol, x, y, font, size, color)
}

// Flyweight factory
type CharacterFactory struct {
    characters map[rune]*Character
    mu         sync.RWMutex
}

func NewCharacterFactory() *CharacterFactory {
    return &CharacterFactory{
        characters: make(map[rune]*Character),
    }
}

func (f *CharacterFactory) GetCharacter(symbol rune) *Character {
    f.mu.RLock()
    if char, exists := f.characters[symbol]; exists {
        f.mu.RUnlock()
        return char
    }
    f.mu.RUnlock()

    f.mu.Lock()
    defer f.mu.Unlock()

    // Double-check after acquiring write lock
    if char, exists := f.characters[symbol]; exists {
        return char
    }

    // Create new character flyweight
    char := &Character{Symbol: symbol}
    f.characters[symbol] = char

    fmt.Printf("Created new character flyweight for '%c'\n", symbol)
    return char
}

func (f *CharacterFactory) GetCreatedCharactersCount() int {
    f.mu.RLock()
    defer f.mu.RUnlock()
    return len(f.characters)
}

func (f *CharacterFactory) ListCreatedCharacters() []rune {
    f.mu.RLock()
    defer f.mu.RUnlock()

    chars := make([]rune, 0, len(f.characters))
    for char := range f.characters {
        chars = append(chars, char)
    }
    return chars
}

// Context - contains extrinsic state
type CharacterContext struct {
    Character *Character // Reference to flyweight
    X         int        // Position X - extrinsic state
    Y         int        // Position Y - extrinsic state
    Font      string     // Font name - extrinsic state
    Size      int        // Font size - extrinsic state
    Color     string     // Color - extrinsic state
}

func NewCharacterContext(char *Character, x, y int, font string, size int, color string) *CharacterContext {
    return &CharacterContext{
        Character: char,
        X:         x,
        Y:         y,
        Font:      font,
        Size:      size,
        Color:     color,
    }
}

func (cc *CharacterContext) Render() string {
    return cc.Character.Render(cc.X, cc.Y, cc.Font, cc.Size, cc.Color)
}

// Document that uses flyweights
type Document struct {
    characters []*CharacterContext
    factory    *CharacterFactory
}

func NewDocument(factory *CharacterFactory) *Document {
    return &Document{
        characters: make([]*CharacterContext, 0),
        factory:    factory,
    }
}

func (d *Document) AddText(text string, x, y int, font string, size int, color string) {
    currentX := x
    for _, symbol := range text {
        if symbol == ' ' {
            currentX += size / 2 // Space width
            continue
        }

        char := d.factory.GetCharacter(symbol)
        context := NewCharacterContext(char, currentX, y, font, size, color)
        d.characters = append(d.characters, context)

        currentX += size // Move to next position
    }
}

func (d *Document) Render() string {
    var result strings.Builder
    result.WriteString("Document Rendering:\n")

    for i, charContext := range d.characters {
        result.WriteString(fmt.Sprintf("%d: %s\n", i+1, charContext.Render()))
    }

    return result.String()
}

func (d *Document) GetCharacterCount() int {
    return len(d.characters)
}

func (d *Document) GetUniqueCharacterCount() int {
    return d.factory.GetCreatedCharactersCount()
}

func (d *Document) GetMemoryUsage() string {
    totalCharacters := d.GetCharacterCount()
    uniqueCharacters := d.GetUniqueCharacterCount()

    // Estimate memory usage
    // Without flyweight: each character context would store the symbol
    memoryWithoutFlyweight := totalCharacters * (4 + 4 + 4 + 20 + 4 + 20) // Rough estimate in bytes

    // With flyweight: shared symbols + contexts with references
    memoryWithFlyweight := (uniqueCharacters * 4) + (totalCharacters * (8 + 4 + 4 + 20 + 4 + 20)) // Rough estimate

    savedMemory := memoryWithoutFlyweight - memoryWithFlyweight
    savingPercentage := float64(savedMemory) / float64(memoryWithoutFlyweight) * 100

    return fmt.Sprintf("Total characters: %d, Unique: %d, Memory saved: ~%d bytes (%.1f%%)",
        totalCharacters, uniqueCharacters, savedMemory, savingPercentage)
}

// Advanced example: Font flyweight with more complex intrinsic state
type FontFlyweight interface {
    RenderText(text string, x, y int, size int, color string) string
    GetFontMetrics(size int) FontMetrics
}

type FontMetrics struct {
    LineHeight int
    Ascent     int
    Descent    int
}

type Font struct {
    FontFamily string
    Style      string    // Bold, Italic, Normal
    Weight     int       // 100-900
    Metrics    map[int]FontMetrics // Size -> Metrics mapping
}

func (f *Font) RenderText(text string, x, y int, size int, color string) string {
    return fmt.Sprintf("Rendering '%s' at (%d,%d) with %s %s %d, size=%d, color=%s",
        text, x, y, f.FontFamily, f.Style, f.Weight, size, color)
}

func (f *Font) GetFontMetrics(size int) FontMetrics {
    if metrics, exists := f.Metrics[size]; exists {
        return metrics
    }

    // Calculate metrics based on size
    return FontMetrics{
        LineHeight: size + size/4,
        Ascent:     size * 3/4,
        Descent:    size / 4,
    }
}

type FontFactory struct {
    fonts map[string]*Font
    mu    sync.RWMutex
}

func NewFontFactory() *FontFactory {
    return &FontFactory{
        fonts: make(map[string]*Font),
    }
}

func (ff *FontFactory) GetFont(family, style string, weight int) *Font {
    key := fmt.Sprintf("%s-%s-%d", family, style, weight)

    ff.mu.RLock()
    if font, exists := ff.fonts[key]; exists {
        ff.mu.RUnlock()
        return font
    }
    ff.mu.RUnlock()

    ff.mu.Lock()
    defer ff.mu.Unlock()

    // Double-check after acquiring write lock
    if font, exists := ff.fonts[key]; exists {
        return font
    }

    // Create new font flyweight
    font := &Font{
        FontFamily: family,
        Style:      style,
        Weight:     weight,
        Metrics:    make(map[int]FontMetrics),
    }

    ff.fonts[key] = font
    fmt.Printf("Created new font flyweight: %s\n", key)
    return font
}

func (ff *FontFactory) GetLoadedFontsCount() int {
    ff.mu.RLock()
    defer ff.mu.RUnlock()
    return len(ff.fonts)
}

// Text element with extrinsic state
type TextElement struct {
    Text     string
    Font     *Font  // Flyweight reference
    X        int
    Y        int
    Size     int
    Color    string
}

func NewTextElement(text string, font *Font, x, y, size int, color string) *TextElement {
    return &TextElement{
        Text:  text,
        Font:  font,
        X:     x,
        Y:     y,
        Size:  size,
        Color: color,
    }
}

func (te *TextElement) Render() string {
    return te.Font.RenderText(te.Text, te.X, te.Y, te.Size, te.Color)
}

func (te *TextElement) GetBounds() (width, height int) {
    metrics := te.Font.GetFontMetrics(te.Size)
    return len(te.Text) * te.Size * 3/5, metrics.LineHeight // Rough estimation
}

// Rich text document
type RichTextDocument struct {
    elements    []*TextElement
    fontFactory *FontFactory
}

func NewRichTextDocument(fontFactory *FontFactory) *RichTextDocument {
    return &RichTextDocument{
        elements:    make([]*TextElement, 0),
        fontFactory: fontFactory,
    }
}

func (rtd *RichTextDocument) AddText(text, fontFamily, style string, weight, x, y, size int, color string) {
    font := rtd.fontFactory.GetFont(fontFamily, style, weight)
    element := NewTextElement(text, font, x, y, size, color)
    rtd.elements = append(rtd.elements, element)
}

func (rtd *RichTextDocument) Render() string {
    var result strings.Builder
    result.WriteString("Rich Text Document:\n")

    for i, element := range rtd.elements {
        result.WriteString(fmt.Sprintf("%d: %s\n", i+1, element.Render()))
    }

    return result.String()
}

func (rtd *RichTextDocument) GetStatistics() string {
    totalElements := len(rtd.elements)
    uniqueFonts := rtd.fontFactory.GetLoadedFontsCount()

    return fmt.Sprintf("Total text elements: %d, Unique fonts loaded: %d", totalElements, uniqueFonts)
}

// Performance demonstration
func demonstratePerformanceBenefits() {
    fmt.Println("\n=== Performance Benefits Demonstration ===")

    factory := NewCharacterFactory()
    document := NewDocument(factory)

    // Add a lot of repeated text
    texts := []string{
        "Hello World! This is a test document.",
        "The quick brown fox jumps over the lazy dog.",
        "Flyweight pattern helps save memory when dealing with large numbers of similar objects.",
        "Hello World! This is a test document.", // Repeated
        "The quick brown fox jumps over the lazy dog.", // Repeated
    }

    y := 10
    for i, text := range texts {
        document.AddText(text, 10, y, "Arial", 12, fmt.Sprintf("Color%d", i%3))
        y += 20
    }

    fmt.Printf("Document created with multiple text lines\n")
    fmt.Printf("Characters in document: %d\n", document.GetCharacterCount())
    fmt.Printf("Unique character flyweights created: %d\n", document.GetUniqueCharacterCount())
    fmt.Printf("Characters created as flyweights: %v\n", document.factory.ListCreatedCharacters())
    fmt.Printf("%s\n", document.GetMemoryUsage())
}

// Stress test to show memory efficiency
func stressTest() {
    fmt.Println("\n=== Stress Test ===")

    factory := NewCharacterFactory()
    document := NewDocument(factory)

    // Add the same text many times
    text := "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    start := time.Now()

    for i := 0; i < 1000; i++ {
        document.AddText(text, 10, i*15, "Arial", 12, "Black")
    }

    elapsed := time.Since(start)

    fmt.Printf("Added text 1000 times (26,000 characters total)\n")
    fmt.Printf("Time taken: %v\n", elapsed)
    fmt.Printf("Unique character flyweights: %d\n", document.GetUniqueCharacterCount())
    fmt.Printf("Total character contexts: %d\n", document.GetCharacterCount())
    fmt.Printf("%s\n", document.GetMemoryUsage())
}

func main() {
    fmt.Println("=== Flyweight Pattern Demo ===\n")

    // Basic flyweight demonstration
    factory := NewCharacterFactory()
    document := NewDocument(factory)

    // Add some text to the document
    document.AddText("Hello", 10, 10, "Arial", 14, "Blue")
    document.AddText("World", 100, 10, "Times", 16, "Red")
    document.AddText("Hello", 10, 30, "Arial", 14, "Green") // Reuses 'H', 'e', 'l', 'o'

    fmt.Println("=== Basic Character Flyweight ===")
    fmt.Printf("Characters created: %d\n", factory.GetCreatedCharactersCount())
    fmt.Printf("Total character instances in document: %d\n", document.GetCharacterCount())

    // Show that flyweights are reused
    h1 := factory.GetCharacter('H')
    h2 := factory.GetCharacter('H')
    fmt.Printf("Same flyweight instance for 'H': %t\n", h1 == h2)

    // Render part of the document
    fmt.Println("\nRendering first few characters:")
    for i := 0; i < min(5, document.GetCharacterCount()); i++ {
        fmt.Printf("  %s\n", document.characters[i].Render())
    }

    // Rich text document demonstration
    fmt.Println("\n=== Rich Text Document with Font Flyweights ===")

    fontFactory := NewFontFactory()
    richDoc := NewRichTextDocument(fontFactory)

    // Add text with different fonts
    richDoc.AddText("Title", "Arial", "Bold", 700, 10, 10, 24, "Black")
    richDoc.AddText("Subtitle", "Arial", "Normal", 400, 10, 40, 18, "Gray")
    richDoc.AddText("Body text", "Times", "Normal", 400, 10, 70, 12, "Black")
    richDoc.AddText("Important note", "Arial", "Bold", 700, 10, 90, 14, "Red") // Reuses Arial Bold

    fmt.Printf("%s\n", richDoc.GetStatistics())

    // Show font reuse
    font1 := fontFactory.GetFont("Arial", "Bold", 700)
    font2 := fontFactory.GetFont("Arial", "Bold", 700)
    fmt.Printf("Same font flyweight instance: %t\n", font1 == font2)

    fmt.Println("\nRich document rendering:")
    fmt.Print(richDoc.Render())

    // Demonstrate performance benefits
    demonstratePerformanceBenefits()

    // Stress test
    stressTest()

    fmt.Println("\n=== Flyweight Pattern Demo Complete ===")
}

// Helper function
func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

// Utility function for ID generation
func generateTransactionID() string {
    return fmt.Sprintf("TXN-%d", time.Now().UnixNano())
}
```

## Variants & Trade-offs

### Variants

1. **Thread-Safe Flyweight Factory**

```go
type ThreadSafeFlyweightFactory struct {
    flyweights map[string]Flyweight
    mu         sync.RWMutex
}

func (f *ThreadSafeFlyweightFactory) GetFlyweight(key string) Flyweight {
    f.mu.RLock()
    if fw, exists := f.flyweights[key]; exists {
        f.mu.RUnlock()
        return fw
    }
    f.mu.RUnlock()

    f.mu.Lock()
    defer f.mu.Unlock()

    // Double-check locking pattern
    if fw, exists := f.flyweights[key]; exists {
        return fw
    }

    fw := f.createFlyweight(key)
    f.flyweights[key] = fw
    return fw
}
```

2. **Lazy Initialization Flyweight**

```go
type LazyFlyweight struct {
    key  string
    data interface{}
    once sync.Once
}

func (lf *LazyFlyweight) GetData() interface{} {
    lf.once.Do(func() {
        lf.data = lf.loadData(lf.key)
    })
    return lf.data
}

func (lf *LazyFlyweight) loadData(key string) interface{} {
    // Expensive data loading operation
    time.Sleep(100 * time.Millisecond)
    return fmt.Sprintf("Data for %s", key)
}
```

3. **Pooled Flyweight Factory**

```go
type PooledFlyweightFactory struct {
    pool     sync.Pool
    active   map[string]Flyweight
    mu       sync.RWMutex
}

func NewPooledFlyweightFactory() *PooledFlyweightFactory {
    return &PooledFlyweightFactory{
        pool: sync.Pool{
            New: func() interface{} {
                return &ConcreteeFlyweight{}
            },
        },
        active: make(map[string]Flyweight),
    }
}

func (f *PooledFlyweightFactory) GetFlyweight(key string) Flyweight {
    f.mu.RLock()
    if fw, exists := f.active[key]; exists {
        f.mu.RUnlock()
        return fw
    }
    f.mu.RUnlock()

    f.mu.Lock()
    defer f.mu.Unlock()

    fw := f.pool.Get().(Flyweight)
    fw.Initialize(key)
    f.active[key] = fw
    return fw
}

func (f *PooledFlyweightFactory) ReleaseFlyweight(key string) {
    f.mu.Lock()
    defer f.mu.Unlock()

    if fw, exists := f.active[key]; exists {
        delete(f.active, key)
        fw.Reset()
        f.pool.Put(fw)
    }
}
```

### Trade-offs

**Pros:**

- **Memory Efficiency**: Significant memory savings with many similar objects
- **Performance**: Reduced object creation overhead
- **Sharing**: Efficient sharing of common state
- **Scalability**: Better scalability with large object collections
- **Cache Friendly**: Better cache performance due to shared objects

**Cons:**

- **Complexity**: Added complexity in separating intrinsic/extrinsic state
- **Thread Safety**: Concurrent access requires careful synchronization
- **Context Overhead**: Extrinsic state must be passed around
- **Debugging**: Harder to debug due to shared state
- **Memory Overhead**: Factory overhead for small object collections

**When to Choose Flyweight vs Alternatives:**

| Scenario              | Pattern         | Reason                  |
| --------------------- | --------------- | ----------------------- |
| Many similar objects  | Flyweight       | Memory efficiency       |
| Unique objects        | Regular objects | No sharing benefit      |
| Immutable shared data | Singleton       | Global shared state     |
| Object pooling        | Object Pool     | Reuse expensive objects |
| Caching               | Cache pattern   | Temporary storage       |

## Integration Tips

### 1. Factory Pattern Integration

```go
type ConfigurableFlyweightFactory struct {
    factories map[string]FlyweightFactory
    config    *FactoryConfig
}

type FactoryConfig struct {
    MaxCacheSize     int
    EnableLazyLoading bool
    ThreadSafe       bool
}

func (cff *ConfigurableFlyweightFactory) GetFactory(factoryType string) FlyweightFactory {
    if factory, exists := cff.factories[factoryType]; exists {
        return factory
    }

    return cff.createFactory(factoryType, cff.config)
}

func (cff *ConfigurableFlyweightFactory) createFactory(factoryType string, config *FactoryConfig) FlyweightFactory {
    switch factoryType {
    case "thread_safe":
        return NewThreadSafeFlyweightFactory()
    case "pooled":
        return NewPooledFlyweightFactory()
    default:
        return NewBasicFlyweightFactory()
    }
}
```

### 2. Observer Pattern Integration

```go
type ObservableFlyweightFactory struct {
    flyweights map[string]Flyweight
    observers  []FactoryObserver
    mu         sync.RWMutex
}

type FactoryObserver interface {
    OnFlyweightCreated(key string, flyweight Flyweight)
    OnFlyweightAccessed(key string, flyweight Flyweight)
}

func (off *ObservableFlyweightFactory) GetFlyweight(key string) Flyweight {
    off.mu.RLock()
    if fw, exists := off.flyweights[key]; exists {
        off.mu.RUnlock()
        off.notifyAccessed(key, fw)
        return fw
    }
    off.mu.RUnlock()

    off.mu.Lock()
    defer off.mu.Unlock()

    fw := off.createFlyweight(key)
    off.flyweights[key] = fw
    off.notifyCreated(key, fw)
    return fw
}

func (off *ObservableFlyweightFactory) notifyCreated(key string, fw Flyweight) {
    for _, observer := range off.observers {
        observer.OnFlyweightCreated(key, fw)
    }
}
```

### 3. Strategy Pattern Integration

```go
type FlyweightCreationStrategy interface {
    CreateFlyweight(key string) Flyweight
    ShouldCache(key string) bool
}

type StandardCreationStrategy struct{}

func (s *StandardCreationStrategy) CreateFlyweight(key string) Flyweight {
    return &ConcreteFlyweight{intrinsicState: key}
}

func (s *StandardCreationStrategy) ShouldCache(key string) bool {
    return true
}

type ConditionalCreationStrategy struct {
    cacheCondition func(string) bool
}

func (c *ConditionalCreationStrategy) ShouldCache(key string) bool {
    return c.cacheCondition(key)
}

type StrategyBasedFactory struct {
    strategy   FlyweightCreationStrategy
    flyweights map[string]Flyweight
    mu         sync.RWMutex
}

func (sbf *StrategyBasedFactory) GetFlyweight(key string) Flyweight {
    if sbf.strategy.ShouldCache(key) {
        return sbf.getCachedFlyweight(key)
    }
    return sbf.strategy.CreateFlyweight(key)
}
```

### 4. Decorator Pattern Integration

```go
type FlyweightDecorator interface {
    Flyweight
    GetDecoratedFlyweight() Flyweight
}

type CachingFlyweightDecorator struct {
    flyweight Flyweight
    cache     map[string]interface{}
    mu        sync.RWMutex
}

func (cfd *CachingFlyweightDecorator) Operation(extrinsicState interface{}) interface{} {
    key := fmt.Sprintf("%v", extrinsicState)

    cfd.mu.RLock()
    if result, exists := cfd.cache[key]; exists {
        cfd.mu.RUnlock()
        return result
    }
    cfd.mu.RUnlock()

    result := cfd.flyweight.Operation(extrinsicState)

    cfd.mu.Lock()
    cfd.cache[key] = result
    cfd.mu.Unlock()

    return result
}

func (cfd *CachingFlyweightDecorator) GetDecoratedFlyweight() Flyweight {
    return cfd.flyweight
}
```

## Common Interview Questions

### 1. **How does Flyweight pattern differ from Singleton pattern?**

**Answer:**
Both patterns involve sharing, but they serve different purposes and have different scopes:

**Flyweight:**

```go
// Multiple flyweight instances, each representing different intrinsic state
type CharacterFlyweight struct {
    symbol rune // Intrinsic state - different for each flyweight
}

type CharacterFactory struct {
    characters map[rune]*CharacterFlyweight // Multiple instances
}

func (f *CharacterFactory) GetCharacter(symbol rune) *CharacterFlyweight {
    // Returns specific flyweight for each symbol
    if char, exists := f.characters[symbol]; exists {
        return char
    }

    char := &CharacterFlyweight{symbol: symbol}
    f.characters[symbol] = char
    return char
}

// Usage: Different flyweights for different symbols
charA := factory.GetCharacter('A') // Flyweight for 'A'
charB := factory.GetCharacter('B') // Different flyweight for 'B'
```

**Singleton:**

```go
// Single instance for entire application
type DatabaseConnection struct {
    connectionString string
}

var (
    instance *DatabaseConnection
    once     sync.Once
)

func GetDatabaseConnection() *DatabaseConnection {
    once.Do(func() {
        instance = &DatabaseConnection{
            connectionString: "database://localhost:5432",
        }
    })
    return instance // Always returns the same instance
}

// Usage: Same instance always
conn1 := GetDatabaseConnection()
conn2 := GetDatabaseConnection()
// conn1 == conn2 (same instance)
```

**Key Differences:**

| Aspect        | Flyweight                                | Singleton             |
| ------------- | ---------------------------------------- | --------------------- |
| **Instances** | Multiple (one per intrinsic state)       | Single instance       |
| **Purpose**   | Memory efficiency                        | Global access point   |
| **State**     | Intrinsic (shared) + Extrinsic (context) | Global state          |
| **Factory**   | Maps keys to instances                   | Returns same instance |
| **Use Case**  | Many similar objects                     | Global resource       |

### 2. **How do you handle thread safety in Flyweight factories?**

**Answer:**
Thread safety in flyweight factories requires careful synchronization to prevent race conditions:

**Double-Checked Locking Pattern:**

```go
type ThreadSafeFlyweightFactory struct {
    flyweights map[string]Flyweight
    mu         sync.RWMutex
}

func (f *ThreadSafeFlyweightFactory) GetFlyweight(key string) Flyweight {
    // First check with read lock (fast path)
    f.mu.RLock()
    if fw, exists := f.flyweights[key]; exists {
        f.mu.RUnlock()
        return fw
    }
    f.mu.RUnlock()

    // Acquire write lock for creation
    f.mu.Lock()
    defer f.mu.Unlock()

    // Double-check after acquiring write lock
    if fw, exists := f.flyweights[key]; exists {
        return fw
    }

    // Create new flyweight
    fw := f.createFlyweight(key)
    f.flyweights[key] = fw
    return fw
}
```

**sync.Once for Lazy Initialization:**

```go
type LazyFlyweight struct {
    key        string
    data       interface{}
    initOnce   sync.Once
    initFunc   func(string) interface{}
}

func (lf *LazyFlyweight) GetData() interface{} {
    lf.initOnce.Do(func() {
        lf.data = lf.initFunc(lf.key)
    })
    return lf.data
}

func NewLazyFlyweight(key string, initFunc func(string) interface{}) *LazyFlyweight {
    return &LazyFlyweight{
        key:      key,
        initFunc: initFunc,
    }
}
```

**Concurrent Map with sync.Map:**

```go
type ConcurrentFlyweightFactory struct {
    flyweights sync.Map // Built-in concurrent map
}

func (f *ConcurrentFlyweightFactory) GetFlyweight(key string) Flyweight {
    if fw, ok := f.flyweights.Load(key); ok {
        return fw.(Flyweight)
    }

    // Create new flyweight
    newFW := f.createFlyweight(key)

    // Store if not already exists
    actual, loaded := f.flyweights.LoadOrStore(key, newFW)
    if loaded {
        // Another goroutine created it first, discard our creation
        return actual.(Flyweight)
    }

    return newFW
}
```

**Channel-Based Factory:**

```go
type ChannelBasedFactory struct {
    requests  chan flyweightRequest
    responses map[string]chan Flyweight
    mu        sync.Mutex
}

type flyweightRequest struct {
    key      string
    response chan Flyweight
}

func (f *ChannelBasedFactory) Start() {
    flyweights := make(map[string]Flyweight)

    go func() {
        for req := range f.requests {
            if fw, exists := flyweights[req.key]; exists {
                req.response <- fw
            } else {
                fw := f.createFlyweight(req.key)
                flyweights[req.key] = fw
                req.response <- fw
            }
        }
    }()
}

func (f *ChannelBasedFactory) GetFlyweight(key string) Flyweight {
    response := make(chan Flyweight, 1)

    f.requests <- flyweightRequest{
        key:      key,
        response: response,
    }

    return <-response
}
```

### 3. **How do you optimize memory usage with Flyweight pattern?**

**Answer:**
Memory optimization in flyweight pattern involves several strategies:

**Efficient Intrinsic State Storage:**

```go
// Optimize storage of intrinsic state
type OptimizedCharacterFlyweight struct {
    symbol    rune   // 4 bytes
    category  uint8  // 1 byte instead of string
    width     uint8  // 1 byte instead of int
    // Total: 6 bytes vs potential 20+ bytes with strings
}

// Use byte constants for categories
const (
    CategoryLetter uint8 = iota
    CategoryDigit
    CategoryPunctuation
    CategorySymbol
)

func (ocf *OptimizedCharacterFlyweight) GetCategory() string {
    categories := []string{"Letter", "Digit", "Punctuation", "Symbol"}
    if int(ocf.category) < len(categories) {
        return categories[ocf.category]
    }
    return "Unknown"
}
```

**Weak References and Cleanup:**

```go
type WeakReferenceFactory struct {
    flyweights map[string]*WeakReference
    mu         sync.RWMutex
    cleaner    *time.Ticker
}

type WeakReference struct {
    flyweight Flyweight
    lastAccess time.Time
    accessCount int64
}

func (wrf *WeakReferenceFactory) GetFlyweight(key string) Flyweight {
    wrf.mu.Lock()
    defer wrf.mu.Unlock()

    if ref, exists := wrf.flyweights[key]; exists {
        ref.lastAccess = time.Now()
        atomic.AddInt64(&ref.accessCount, 1)
        return ref.flyweight
    }

    fw := wrf.createFlyweight(key)
    wrf.flyweights[key] = &WeakReference{
        flyweight:   fw,
        lastAccess:  time.Now(),
        accessCount: 1,
    }

    return fw
}

func (wrf *WeakReferenceFactory) startCleaner() {
    wrf.cleaner = time.NewTicker(5 * time.Minute)

    go func() {
        for range wrf.cleaner.C {
            wrf.cleanup()
        }
    }()
}

func (wrf *WeakReferenceFactory) cleanup() {
    wrf.mu.Lock()
    defer wrf.mu.Unlock()

    cutoff := time.Now().Add(-10 * time.Minute)

    for key, ref := range wrf.flyweights {
        if ref.lastAccess.Before(cutoff) && ref.accessCount < 10 {
            delete(wrf.flyweights, key)
        }
    }
}
```

**Memory Pool Integration:**

```go
type PooledFlyweightFactory struct {
    pool       sync.Pool
    active     map[string]Flyweight
    mu         sync.RWMutex
    maxActive  int
}

func NewPooledFlyweightFactory(maxActive int) *PooledFlyweightFactory {
    return &PooledFlyweightFactory{
        pool: sync.Pool{
            New: func() interface{} {
                return &ReusableFlyweight{}
            },
        },
        active:    make(map[string]Flyweight),
        maxActive: maxActive,
    }
}

func (pff *PooledFlyweightFactory) GetFlyweight(key string) Flyweight {
    pff.mu.RLock()
    if fw, exists := pff.active[key]; exists {
        pff.mu.RUnlock()
        return fw
    }
    pff.mu.RUnlock()

    pff.mu.Lock()
    defer pff.mu.Unlock()

    if len(pff.active) >= pff.maxActive {
        // Remove least recently used
        pff.evictLRU()
    }

    fw := pff.pool.Get().(Flyweight)
    fw.Initialize(key)
    pff.active[key] = fw

    return fw
}

func (pff *PooledFlyweightFactory) ReleaseFlyweight(key string) {
    pff.mu.Lock()
    defer pff.mu.Unlock()

    if fw, exists := pff.active[key]; exists {
        delete(pff.active, key)
        fw.Reset()
        pff.pool.Put(fw)
    }
}
```

**Compression and Bit Packing:**

```go
// Pack multiple boolean flags into a single byte
type CompactFlyweight struct {
    id    uint32
    flags uint8 // 8 boolean flags packed into 1 byte
}

const (
    FlagVisible uint8 = 1 << iota
    FlagEditable
    FlagRequired
    FlagEnabled
    FlagFocusable
    FlagSelectable
    FlagDraggable
    FlagDroppable
)

func (cf *CompactFlyweight) HasFlag(flag uint8) bool {
    return cf.flags&flag != 0
}

func (cf *CompactFlyweight) SetFlag(flag uint8, value bool) {
    if value {
        cf.flags |= flag
    } else {
        cf.flags &^= flag
    }
}
```

### 4. **How do you test Flyweight pattern implementations?**

**Answer:**
Testing flyweight patterns requires verifying both functionality and memory efficiency:

**Factory Sharing Test:**

```go
func TestFlyweightSharing(t *testing.T) {
    factory := NewCharacterFactory()

    // Get same character multiple times
    char1 := factory.GetCharacter('A')
    char2 := factory.GetCharacter('A')
    char3 := factory.GetCharacter('B')

    // Verify flyweights are shared
    assert.Same(t, char1, char2, "Same character should return same flyweight instance")
    assert.NotSame(t, char1, char3, "Different characters should return different flyweight instances")

    // Verify factory state
    assert.Equal(t, 2, factory.GetCreatedCharactersCount(), "Should have created exactly 2 flyweights")
}
```

**Memory Usage Test:**

```go
func TestMemoryEfficiency(t *testing.T) {
    factory := NewCharacterFactory()
    document := NewDocument(factory)

    // Add large amount of repeated text
    text := "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i := 0; i < 1000; i++ {
        document.AddText(text, 10, i*15, "Arial", 12, "Black")
    }

    // Should have created only 26 flyweights despite 26,000 characters
    assert.Equal(t, 26, factory.GetCreatedCharactersCount(),
        "Should only create flyweights for unique characters")
    assert.Equal(t, 26000, document.GetCharacterCount(),
        "Should have 26,000 character contexts")
}
```

**Thread Safety Test:**

```go
func TestConcurrentAccess(t *testing.T) {
    factory := NewThreadSafeFlyweightFactory()

    var wg sync.WaitGroup
    results := make([]Flyweight, 100)

    // Access same flyweight concurrently
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func(index int) {
            defer wg.Done()
            results[index] = factory.GetFlyweight("test-key")
        }(i)
    }

    wg.Wait()

    // All results should be the same instance
    for i := 1; i < len(results); i++ {
        assert.Same(t, results[0], results[i],
            "Concurrent access should return same flyweight instance")
    }

    // Factory should have created only one instance
    assert.Equal(t, 1, factory.GetCreatedFlyweightsCount())
}
```

**Performance Benchmark:**

```go
func BenchmarkFlyweightVsRegularObjects(b *testing.B) {
    b.Run("WithFlyweight", func(b *testing.B) {
        factory := NewCharacterFactory()

        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            char := factory.GetCharacter('A')
            char.Render(10, 10, "Arial", 12, "Black")
        }
    })

    b.Run("WithoutFlyweight", func(b *testing.B) {
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            char := &Character{Symbol: 'A'} // Create new instance each time
            char.Render(10, 10, "Arial", 12, "Black")
        }
    })
}

func BenchmarkMemoryUsage(b *testing.B) {
    factory := NewCharacterFactory()

    b.ReportAllocs()
    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        // This should not allocate new flyweights after first iteration
        char := factory.GetCharacter('A')
        _ = char
    }
}
```

**Extrinsic State Test:**

```go
func TestExtrinsicStateHandling(t *testing.T) {
    factory := NewCharacterFactory()
    char := factory.GetCharacter('A')

    // Same flyweight should produce different results with different extrinsic state
    result1 := char.Render(10, 10, "Arial", 12, "Red")
    result2 := char.Render(20, 20, "Times", 16, "Blue")

    assert.NotEqual(t, result1, result2,
        "Same flyweight with different extrinsic state should produce different results")

    // Verify flyweight itself doesn't store extrinsic state
    assert.Equal(t, 'A', char.Symbol, "Flyweight should only contain intrinsic state")
}
```

### 5. **When should you not use Flyweight pattern?**

**Answer:**
Flyweight pattern should be avoided in certain scenarios where its costs outweigh benefits:

**Small Number of Objects:**

```go
// DON'T use flyweight for small collections
type SmallConfigManager struct {
    configs [5]*Config // Only 5 configs total
}

// The overhead of flyweight factory isn't worth it
func (scm *SmallConfigManager) GetConfig(index int) *Config {
    return scm.configs[index] // Direct access is better
}
```

**Unique State Dominant:**

```go
// DON'T use flyweight when objects have mostly unique state
type UserProfile struct {
    // Mostly unique/extrinsic state
    UserID      string
    Name        string
    Email       string
    Preferences map[string]string

    // Minimal intrinsic state
    AccountType string // Only this could be shared
}

// Better as regular objects since sharing benefit is minimal
func NewUserProfile(userID, name, email, accountType string) *UserProfile {
    return &UserProfile{
        UserID:      userID,
        Name:        name,
        Email:       email,
        AccountType: accountType,
        Preferences: make(map[string]string),
    }
}
```

**Performance-Critical Code:**

```go
// DON'T use flyweight in tight loops where lookup overhead matters
func ProcessHighFrequencyData(data []DataPoint) {
    for _, point := range data {
        // Factory lookup in tight loop adds overhead
        // processor := factory.GetProcessor(point.Type) // Avoid this

        // Direct processing is faster
        switch point.Type {
        case "TYPE_A":
            processTypeA(point)
        case "TYPE_B":
            processTypeB(point)
        }
    }
}
```

**Complex Thread Safety Requirements:**

```go
// DON'T use flyweight when thread safety adds too much complexity
type ComplexStatefulFlyweight struct {
    intrinsicState string
    cache          map[string]interface{}
    metrics        *Metrics
    mu             sync.RWMutex
}

// Thread safety complexity might outweigh benefits
func (csf *ComplexStatefulFlyweight) Operation(extrinsicState interface{}) interface{} {
    csf.mu.Lock()
    defer csf.mu.Unlock()

    // Complex synchronized operations
    // Better to use separate instances
    return nil
}
```

**Better Alternatives:**

| Scenario                 | Alternative           | Reason                   |
| ------------------------ | --------------------- | ------------------------ |
| Small object count       | Regular objects       | No memory pressure       |
| Mostly unique state      | Value objects         | No sharing benefit       |
| High-performance code    | Direct implementation | Avoid lookup overhead    |
| Complex state management | Separate objects      | Simpler design           |
| Temporary objects        | Object pooling        | Lifecycle management     |
| Immutable data           | Caching               | Different access pattern |

**Decision Framework:**

```go
type FlyweightDecision struct {
    ObjectCount        int
    SharedStateRatio   float64 // 0.0 to 1.0
    MemoryConstraints  bool
    PerformanceNeeds   string // "high", "medium", "low"
    ThreadSafetyNeeds  bool
}

func (fd *FlyweightDecision) ShouldUseFlyweight() bool {
    if fd.ObjectCount < 100 {
        return false // Too few objects
    }

    if fd.SharedStateRatio < 0.3 {
        return false // Not enough shared state
    }

    if fd.PerformanceNeeds == "high" && fd.ThreadSafetyNeeds {
        return false // Synchronization overhead too high
    }

    return fd.MemoryConstraints || fd.ObjectCount > 10000
}
```
