package strategy

import (
	"context"
	"fmt"
	"time"
)

// StripePaymentStrategy implements PaymentStrategy for Stripe
type StripePaymentStrategy struct {
	apiKey    string
	timeout   time.Duration
	available bool
}

// NewStripePaymentStrategy creates a new Stripe payment strategy
func NewStripePaymentStrategy(apiKey string, timeout time.Duration) *StripePaymentStrategy {
	return &StripePaymentStrategy{
		apiKey:    apiKey,
		timeout:   timeout,
		available: true,
	}
}

// ProcessPayment processes payment using Stripe
func (s *StripePaymentStrategy) ProcessPayment(ctx context.Context, request PaymentRequest) (*PaymentResponse, error) {
	// Simulate Stripe payment processing
	time.Sleep(100 * time.Millisecond)

	response := &PaymentResponse{
		PaymentID:     request.PaymentID,
		Status:        "completed",
		TransactionID: fmt.Sprintf("stripe_%s", request.PaymentID),
		Amount:        request.Amount,
		Currency:      request.Currency,
		Gateway:       "stripe",
		ProcessedAt:   time.Now(),
		Metadata:      request.Metadata,
	}

	return response, nil
}

// ValidatePayment validates payment request for Stripe
func (s *StripePaymentStrategy) ValidatePayment(ctx context.Context, request PaymentRequest) error {
	if request.Amount <= 0 {
		return fmt.Errorf("invalid amount: %f", request.Amount)
	}
	if request.Currency != "USD" && request.Currency != "EUR" {
		return fmt.Errorf("unsupported currency: %s", request.Currency)
	}
	return nil
}

// GetStrategyName returns the strategy name
func (s *StripePaymentStrategy) GetStrategyName() string {
	return "stripe"
}

// GetSupportedCurrencies returns supported currencies
func (s *StripePaymentStrategy) GetSupportedCurrencies() []string {
	return []string{"USD", "EUR"}
}

// GetProcessingTime returns processing time
func (s *StripePaymentStrategy) GetProcessingTime() time.Duration {
	return s.timeout
}

// IsAvailable returns availability status
func (s *StripePaymentStrategy) IsAvailable() bool {
	return s.available
}

// RazorpayPaymentStrategy implements PaymentStrategy for Razorpay
type RazorpayPaymentStrategy struct {
	apiKey    string
	timeout   time.Duration
	available bool
}

// NewRazorpayPaymentStrategy creates a new Razorpay payment strategy
func NewRazorpayPaymentStrategy(apiKey string, timeout time.Duration) *RazorpayPaymentStrategy {
	return &RazorpayPaymentStrategy{
		apiKey:    apiKey,
		timeout:   timeout,
		available: true,
	}
}

// ProcessPayment processes payment using Razorpay
func (r *RazorpayPaymentStrategy) ProcessPayment(ctx context.Context, request PaymentRequest) (*PaymentResponse, error) {
	// Simulate Razorpay payment processing
	time.Sleep(150 * time.Millisecond)

	response := &PaymentResponse{
		PaymentID:     request.PaymentID,
		Status:        "completed",
		TransactionID: fmt.Sprintf("razorpay_%s", request.PaymentID),
		Amount:        request.Amount,
		Currency:      request.Currency,
		Gateway:       "razorpay",
		ProcessedAt:   time.Now(),
		Metadata:      request.Metadata,
	}

	return response, nil
}

// ValidatePayment validates payment request for Razorpay
func (r *RazorpayPaymentStrategy) ValidatePayment(ctx context.Context, request PaymentRequest) error {
	if request.Amount <= 0 {
		return fmt.Errorf("invalid amount: %f", request.Amount)
	}
	if request.Currency != "INR" {
		return fmt.Errorf("unsupported currency: %s", request.Currency)
	}
	return nil
}

// GetStrategyName returns the strategy name
func (r *RazorpayPaymentStrategy) GetStrategyName() string {
	return "razorpay"
}

// GetSupportedCurrencies returns supported currencies
func (r *RazorpayPaymentStrategy) GetSupportedCurrencies() []string {
	return []string{"INR"}
}

// GetProcessingTime returns processing time
func (r *RazorpayPaymentStrategy) GetProcessingTime() time.Duration {
	return r.timeout
}

// IsAvailable returns availability status
func (r *RazorpayPaymentStrategy) IsAvailable() bool {
	return r.available
}

// PayPalPaymentStrategy implements PaymentStrategy for PayPal
type PayPalPaymentStrategy struct {
	apiKey    string
	timeout   time.Duration
	available bool
}

// NewPayPalPaymentStrategy creates a new PayPal payment strategy
func NewPayPalPaymentStrategy(apiKey string, timeout time.Duration) *PayPalPaymentStrategy {
	return &PayPalPaymentStrategy{
		apiKey:    apiKey,
		timeout:   timeout,
		available: true,
	}
}

// ProcessPayment processes payment using PayPal
func (p *PayPalPaymentStrategy) ProcessPayment(ctx context.Context, request PaymentRequest) (*PaymentResponse, error) {
	// Simulate PayPal payment processing
	time.Sleep(200 * time.Millisecond)

	response := &PaymentResponse{
		PaymentID:     request.PaymentID,
		Status:        "completed",
		TransactionID: fmt.Sprintf("paypal_%s", request.PaymentID),
		Amount:        request.Amount,
		Currency:      request.Currency,
		Gateway:       "paypal",
		ProcessedAt:   time.Now(),
		Metadata:      request.Metadata,
	}

	return response, nil
}

// ValidatePayment validates payment request for PayPal
func (p *PayPalPaymentStrategy) ValidatePayment(ctx context.Context, request PaymentRequest) error {
	if request.Amount <= 0 {
		return fmt.Errorf("invalid amount: %f", request.Amount)
	}
	if request.Currency != "USD" && request.Currency != "EUR" && request.Currency != "GBP" {
		return fmt.Errorf("unsupported currency: %s", request.Currency)
	}
	return nil
}

// GetStrategyName returns the strategy name
func (p *PayPalPaymentStrategy) GetStrategyName() string {
	return "paypal"
}

// GetSupportedCurrencies returns supported currencies
func (p *PayPalPaymentStrategy) GetSupportedCurrencies() []string {
	return []string{"USD", "EUR", "GBP"}
}

// GetProcessingTime returns processing time
func (p *PayPalPaymentStrategy) GetProcessingTime() time.Duration {
	return p.timeout
}

// IsAvailable returns availability status
func (p *PayPalPaymentStrategy) IsAvailable() bool {
	return p.available
}

// BankTransferPaymentStrategy implements PaymentStrategy for Bank Transfer
type BankTransferPaymentStrategy struct {
	apiKey    string
	timeout   time.Duration
	available bool
}

// NewBankTransferPaymentStrategy creates a new Bank Transfer payment strategy
func NewBankTransferPaymentStrategy(apiKey string, timeout time.Duration) *BankTransferPaymentStrategy {
	return &BankTransferPaymentStrategy{
		apiKey:    apiKey,
		timeout:   timeout,
		available: true,
	}
}

// ProcessPayment processes payment using Bank Transfer
func (b *BankTransferPaymentStrategy) ProcessPayment(ctx context.Context, request PaymentRequest) (*PaymentResponse, error) {
	// Simulate Bank Transfer payment processing
	time.Sleep(300 * time.Millisecond)

	response := &PaymentResponse{
		PaymentID:     request.PaymentID,
		Status:        "pending",
		TransactionID: fmt.Sprintf("bank_%s", request.PaymentID),
		Amount:        request.Amount,
		Currency:      request.Currency,
		Gateway:       "bank_transfer",
		ProcessedAt:   time.Now(),
		Metadata:      request.Metadata,
	}

	return response, nil
}

// ValidatePayment validates payment request for Bank Transfer
func (b *BankTransferPaymentStrategy) ValidatePayment(ctx context.Context, request PaymentRequest) error {
	if request.Amount <= 0 {
		return fmt.Errorf("invalid amount: %f", request.Amount)
	}
	if request.Amount < 100 {
		return fmt.Errorf("minimum amount for bank transfer is 100")
	}
	return nil
}

// GetStrategyName returns the strategy name
func (b *BankTransferPaymentStrategy) GetStrategyName() string {
	return "bank_transfer"
}

// GetSupportedCurrencies returns supported currencies
func (b *BankTransferPaymentStrategy) GetSupportedCurrencies() []string {
	return []string{"USD", "EUR", "GBP", "INR"}
}

// GetProcessingTime returns processing time
func (b *BankTransferPaymentStrategy) GetProcessingTime() time.Duration {
	return b.timeout
}

// IsAvailable returns availability status
func (b *BankTransferPaymentStrategy) IsAvailable() bool {
	return b.available
}
