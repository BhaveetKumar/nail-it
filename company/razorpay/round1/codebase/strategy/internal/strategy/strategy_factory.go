package strategy

import (
	"fmt"
	"time"
)

// StrategyFactoryImpl implements StrategyFactory interface
type StrategyFactoryImpl struct {
	config *StrategyConfig
}

// NewStrategyFactory creates a new strategy factory
func NewStrategyFactory(config *StrategyConfig) *StrategyFactoryImpl {
	return &StrategyFactoryImpl{
		config: config,
	}
}

// CreatePaymentStrategy creates a payment strategy
func (sf *StrategyFactoryImpl) CreatePaymentStrategy(strategyType string) (PaymentStrategy, error) {
	switch strategyType {
	case "stripe":
		return NewStripePaymentStrategy("sk_test_...", 5*time.Second), nil
	case "razorpay":
		return NewRazorpayPaymentStrategy("rzp_test_...", 5*time.Second), nil
	case "paypal":
		return NewPayPalPaymentStrategy("paypal_...", 5*time.Second), nil
	case "bank_transfer":
		return NewBankTransferPaymentStrategy("bank_...", 10*time.Second), nil
	default:
		return nil, fmt.Errorf("unsupported payment strategy type: %s", strategyType)
	}
}

// CreateNotificationStrategy creates a notification strategy
func (sf *StrategyFactoryImpl) CreateNotificationStrategy(strategyType string) (NotificationStrategy, error) {
	switch strategyType {
	case "email":
		return NewEmailNotificationStrategy("smtp.gmail.com", 587, 5*time.Second), nil
	case "sms":
		return NewSMSNotificationStrategy("sms_api_...", 3*time.Second), nil
	case "push":
		return NewPushNotificationStrategy("push_api_...", 2*time.Second), nil
	case "webhook":
		return NewWebhookNotificationStrategy("webhook_...", 5*time.Second), nil
	case "slack":
		return NewSlackNotificationStrategy("slack_...", 3*time.Second), nil
	default:
		return nil, fmt.Errorf("unsupported notification strategy type: %s", strategyType)
	}
}

// CreatePricingStrategy creates a pricing strategy
func (sf *StrategyFactoryImpl) CreatePricingStrategy(strategyType string) (PricingStrategy, error) {
	switch strategyType {
	case "standard":
		return NewStandardPricingStrategy(), nil
	case "discount":
		return NewDiscountPricingStrategy(), nil
	case "dynamic":
		return NewDynamicPricingStrategy(), nil
	case "tiered":
		return NewTieredPricingStrategy(), nil
	default:
		return nil, fmt.Errorf("unsupported pricing strategy type: %s", strategyType)
	}
}

// CreateAuthenticationStrategy creates an authentication strategy
func (sf *StrategyFactoryImpl) CreateAuthenticationStrategy(strategyType string) (AuthenticationStrategy, error) {
	switch strategyType {
	case "jwt":
		return NewJWTAuthenticationStrategy(), nil
	case "oauth":
		return NewOAuthAuthenticationStrategy(), nil
	case "basic":
		return NewBasicAuthenticationStrategy(), nil
	case "api_key":
		return NewAPIKeyAuthenticationStrategy(), nil
	default:
		return nil, fmt.Errorf("unsupported authentication strategy type: %s", strategyType)
	}
}

// CreateCachingStrategy creates a caching strategy
func (sf *StrategyFactoryImpl) CreateCachingStrategy(strategyType string) (CachingStrategy, error) {
	switch strategyType {
	case "redis":
		return NewRedisCachingStrategy(), nil
	case "memory":
		return NewMemoryCachingStrategy(), nil
	case "database":
		return NewDatabaseCachingStrategy(), nil
	case "hybrid":
		return NewHybridCachingStrategy(), nil
	default:
		return nil, fmt.Errorf("unsupported caching strategy type: %s", strategyType)
	}
}

// CreateLoggingStrategy creates a logging strategy
func (sf *StrategyFactoryImpl) CreateLoggingStrategy(strategyType string) (LoggingStrategy, error) {
	switch strategyType {
	case "file":
		return NewFileLoggingStrategy(), nil
	case "console":
		return NewConsoleLoggingStrategy(), nil
	case "database":
		return NewDatabaseLoggingStrategy(), nil
	case "remote":
		return NewRemoteLoggingStrategy(), nil
	default:
		return nil, fmt.Errorf("unsupported logging strategy type: %s", strategyType)
	}
}

// CreateDataProcessingStrategy creates a data processing strategy
func (sf *StrategyFactoryImpl) CreateDataProcessingStrategy(strategyType string) (DataProcessingStrategy, error) {
	switch strategyType {
	case "json":
		return NewJSONDataProcessingStrategy(), nil
	case "xml":
		return NewXMLDataProcessingStrategy(), nil
	case "csv":
		return NewCSVDataProcessingStrategy(), nil
	case "binary":
		return NewBinaryDataProcessingStrategy(), nil
	default:
		return nil, fmt.Errorf("unsupported data processing strategy type: %s", strategyType)
	}
}

// StandardPricingStrategy implements PricingStrategy for standard pricing
type StandardPricingStrategy struct {
	timeout   time.Duration
	available bool
}

// NewStandardPricingStrategy creates a new standard pricing strategy
func NewStandardPricingStrategy() *StandardPricingStrategy {
	return &StandardPricingStrategy{
		timeout:   100 * time.Millisecond,
		available: true,
	}
}

// CalculatePrice calculates standard price
func (s *StandardPricingStrategy) CalculatePrice(ctx context.Context, request PricingRequest) (*PricingResponse, error) {
	time.Sleep(s.timeout)
	
	response := &PricingResponse{
		PricingID:     request.PricingID,
		ProductID:     request.ProductID,
		BasePrice:     request.BasePrice,
		DiscountPrice: 0,
		FinalPrice:    request.BasePrice * float64(request.Quantity),
		Currency:      request.Currency,
		DiscountCode:  request.DiscountCode,
		CalculatedAt:  time.Now(),
		Metadata:      request.Metadata,
	}
	
	return response, nil
}

// ValidatePricing validates pricing request
func (s *StandardPricingStrategy) ValidatePricing(ctx context.Context, request PricingRequest) error {
	if request.BasePrice <= 0 {
		return fmt.Errorf("invalid base price: %f", request.BasePrice)
	}
	if request.Quantity <= 0 {
		return fmt.Errorf("invalid quantity: %d", request.Quantity)
	}
	return nil
}

// GetStrategyName returns the strategy name
func (s *StandardPricingStrategy) GetStrategyName() string {
	return "standard"
}

// GetSupportedProducts returns supported products
func (s *StandardPricingStrategy) GetSupportedProducts() []string {
	return []string{"all"}
}

// GetCalculationTime returns calculation time
func (s *StandardPricingStrategy) GetCalculationTime() time.Duration {
	return s.timeout
}

// IsAvailable returns availability status
func (s *StandardPricingStrategy) IsAvailable() bool {
	return s.available
}

// DiscountPricingStrategy implements PricingStrategy for discount pricing
type DiscountPricingStrategy struct {
	timeout   time.Duration
	available bool
}

// NewDiscountPricingStrategy creates a new discount pricing strategy
func NewDiscountPricingStrategy() *DiscountPricingStrategy {
	return &DiscountPricingStrategy{
		timeout:   150 * time.Millisecond,
		available: true,
	}
}

// CalculatePrice calculates price with discount
func (d *DiscountPricingStrategy) CalculatePrice(ctx context.Context, request PricingRequest) (*PricingResponse, error) {
	time.Sleep(d.timeout)
	
	basePrice := request.BasePrice * float64(request.Quantity)
	discountPrice := 0.0
	
	// Apply discount based on discount code
	if request.DiscountCode != "" {
		switch request.DiscountCode {
		case "SAVE10":
			discountPrice = basePrice * 0.1
		case "SAVE20":
			discountPrice = basePrice * 0.2
		case "SAVE50":
			discountPrice = basePrice * 0.5
		}
	}
	
	finalPrice := basePrice - discountPrice
	
	response := &PricingResponse{
		PricingID:     request.PricingID,
		ProductID:     request.ProductID,
		BasePrice:     basePrice,
		DiscountPrice: discountPrice,
		FinalPrice:    finalPrice,
		Currency:      request.Currency,
		DiscountCode:  request.DiscountCode,
		CalculatedAt:  time.Now(),
		Metadata:      request.Metadata,
	}
	
	return response, nil
}

// ValidatePricing validates pricing request
func (d *DiscountPricingStrategy) ValidatePricing(ctx context.Context, request PricingRequest) error {
	if request.BasePrice <= 0 {
		return fmt.Errorf("invalid base price: %f", request.BasePrice)
	}
	if request.Quantity <= 0 {
		return fmt.Errorf("invalid quantity: %d", request.Quantity)
	}
	return nil
}

// GetStrategyName returns the strategy name
func (d *DiscountPricingStrategy) GetStrategyName() string {
	return "discount"
}

// GetSupportedProducts returns supported products
func (d *DiscountPricingStrategy) GetSupportedProducts() []string {
	return []string{"all"}
}

// GetCalculationTime returns calculation time
func (d *DiscountPricingStrategy) GetCalculationTime() time.Duration {
	return d.timeout
}

// IsAvailable returns availability status
func (d *DiscountPricingStrategy) IsAvailable() bool {
	return d.available
}

// DynamicPricingStrategy implements PricingStrategy for dynamic pricing
type DynamicPricingStrategy struct {
	timeout   time.Duration
	available bool
}

// NewDynamicPricingStrategy creates a new dynamic pricing strategy
func NewDynamicPricingStrategy() *DynamicPricingStrategy {
	return &DynamicPricingStrategy{
		timeout:   200 * time.Millisecond,
		available: true,
	}
}

// CalculatePrice calculates dynamic price
func (d *DynamicPricingStrategy) CalculatePrice(ctx context.Context, request PricingRequest) (*PricingResponse, error) {
	time.Sleep(d.timeout)
	
	basePrice := request.BasePrice * float64(request.Quantity)
	
	// Apply dynamic pricing based on time, demand, etc.
	multiplier := 1.0
	hour := time.Now().Hour()
	
	if hour >= 9 && hour <= 17 {
		multiplier = 1.2 // Peak hours
	} else if hour >= 18 && hour <= 22 {
		multiplier = 1.1 // Evening hours
	} else {
		multiplier = 0.9 // Off-peak hours
	}
	
	finalPrice := basePrice * multiplier
	
	response := &PricingResponse{
		PricingID:     request.PricingID,
		ProductID:     request.ProductID,
		BasePrice:     basePrice,
		DiscountPrice: basePrice - finalPrice,
		FinalPrice:    finalPrice,
		Currency:      request.Currency,
		DiscountCode:  request.DiscountCode,
		CalculatedAt:  time.Now(),
		Metadata:      request.Metadata,
	}
	
	return response, nil
}

// ValidatePricing validates pricing request
func (d *DynamicPricingStrategy) ValidatePricing(ctx context.Context, request PricingRequest) error {
	if request.BasePrice <= 0 {
		return fmt.Errorf("invalid base price: %f", request.BasePrice)
	}
	if request.Quantity <= 0 {
		return fmt.Errorf("invalid quantity: %d", request.Quantity)
	}
	return nil
}

// GetStrategyName returns the strategy name
func (d *DynamicPricingStrategy) GetStrategyName() string {
	return "dynamic"
}

// GetSupportedProducts returns supported products
func (d *DynamicPricingStrategy) GetSupportedProducts() []string {
	return []string{"all"}
}

// GetCalculationTime returns calculation time
func (d *DynamicPricingStrategy) GetCalculationTime() time.Duration {
	return d.timeout
}

// IsAvailable returns availability status
func (d *DynamicPricingStrategy) IsAvailable() bool {
	return d.available
}

// TieredPricingStrategy implements PricingStrategy for tiered pricing
type TieredPricingStrategy struct {
	timeout   time.Duration
	available bool
}

// NewTieredPricingStrategy creates a new tiered pricing strategy
func NewTieredPricingStrategy() *TieredPricingStrategy {
	return &TieredPricingStrategy{
		timeout:   180 * time.Millisecond,
		available: true,
	}
}

// CalculatePrice calculates tiered price
func (t *TieredPricingStrategy) CalculatePrice(ctx context.Context, request PricingRequest) (*PricingResponse, error) {
	time.Sleep(t.timeout)
	
	basePrice := request.BasePrice * float64(request.Quantity)
	
	// Apply tiered pricing
	var finalPrice float64
	quantity := request.Quantity
	
	if quantity <= 10 {
		finalPrice = basePrice
	} else if quantity <= 50 {
		finalPrice = basePrice * 0.95 // 5% discount
	} else if quantity <= 100 {
		finalPrice = basePrice * 0.90 // 10% discount
	} else {
		finalPrice = basePrice * 0.85 // 15% discount
	}
	
	response := &PricingResponse{
		PricingID:     request.PricingID,
		ProductID:     request.ProductID,
		BasePrice:     basePrice,
		DiscountPrice: basePrice - finalPrice,
		FinalPrice:    finalPrice,
		Currency:      request.Currency,
		DiscountCode:  request.DiscountCode,
		CalculatedAt:  time.Now(),
		Metadata:      request.Metadata,
	}
	
	return response, nil
}

// ValidatePricing validates pricing request
func (t *TieredPricingStrategy) ValidatePricing(ctx context.Context, request PricingRequest) error {
	if request.BasePrice <= 0 {
		return fmt.Errorf("invalid base price: %f", request.BasePrice)
	}
	if request.Quantity <= 0 {
		return fmt.Errorf("invalid quantity: %d", request.Quantity)
	}
	return nil
}

// GetStrategyName returns the strategy name
func (t *TieredPricingStrategy) GetStrategyName() string {
	return "tiered"
}

// GetSupportedProducts returns supported products
func (t *TieredPricingStrategy) GetSupportedProducts() []string {
	return []string{"all"}
}

// GetCalculationTime returns calculation time
func (t *TieredPricingStrategy) GetCalculationTime() time.Duration {
	return t.timeout
}

// IsAvailable returns availability status
func (t *TieredPricingStrategy) IsAvailable() bool {
	return t.available
}
