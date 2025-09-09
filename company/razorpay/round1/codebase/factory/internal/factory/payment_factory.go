package factory

import (
	"context"
	"fmt"
	"sync"

	"factory-service/internal/config"
	"factory-service/internal/logger"
	"factory-service/internal/models"
)

// PaymentGateway interface defines the contract for payment gateways
type PaymentGateway interface {
	ProcessPayment(ctx context.Context, request *models.PaymentRequest) (*models.PaymentResponse, error)
	RefundPayment(ctx context.Context, request *models.RefundRequest) (*models.RefundResponse, error)
	GetPaymentStatus(ctx context.Context, paymentID string) (*models.PaymentStatus, error)
	ValidatePayment(request *models.PaymentRequest) error
	GetGatewayName() string
}

// PaymentGatewayFactory implements the Factory pattern for creating payment gateways
type PaymentGatewayFactory struct {
	gateways map[string]func() PaymentGateway
	mutex    sync.RWMutex
}

var (
	paymentGatewayFactory *PaymentGatewayFactory
	factoryOnce           sync.Once
)

// GetPaymentGatewayFactory returns the singleton instance of PaymentGatewayFactory
func GetPaymentGatewayFactory() *PaymentGatewayFactory {
	factoryOnce.Do(func() {
		paymentGatewayFactory = &PaymentGatewayFactory{
			gateways: make(map[string]func() PaymentGateway),
		}
		paymentGatewayFactory.registerDefaultGateways()
	})
	return paymentGatewayFactory
}

// registerDefaultGateways registers the default payment gateways
func (pgf *PaymentGatewayFactory) registerDefaultGateways() {
	pgf.mutex.Lock()
	defer pgf.mutex.Unlock()

	// Register Stripe gateway
	pgf.gateways["stripe"] = func() PaymentGateway {
		return NewStripeGateway()
	}

	// Register PayPal gateway
	pgf.gateways["paypal"] = func() PaymentGateway {
		return NewPayPalGateway()
	}

	// Register Razorpay gateway
	pgf.gateways["razorpay"] = func() PaymentGateway {
		return NewRazorpayGateway()
	}

	// Register Bank Transfer gateway
	pgf.gateways["bank_transfer"] = func() PaymentGateway {
		return NewBankTransferGateway()
	}

	// Register Digital Wallet gateway
	pgf.gateways["digital_wallet"] = func() PaymentGateway {
		return NewDigitalWalletGateway()
	}
}

// RegisterGateway registers a new payment gateway
func (pgf *PaymentGatewayFactory) RegisterGateway(name string, creator func() PaymentGateway) {
	pgf.mutex.Lock()
	defer pgf.mutex.Unlock()
	pgf.gateways[name] = creator
}

// CreateGateway creates a payment gateway instance
func (pgf *PaymentGatewayFactory) CreateGateway(gatewayType string) (PaymentGateway, error) {
	pgf.mutex.RLock()
	creator, exists := pgf.gateways[gatewayType]
	pgf.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("payment gateway type '%s' not supported", gatewayType)
	}

	return creator(), nil
}

// GetAvailableGateways returns the list of available gateway types
func (pgf *PaymentGatewayFactory) GetAvailableGateways() []string {
	pgf.mutex.RLock()
	defer pgf.mutex.RUnlock()

	gateways := make([]string, 0, len(pgf.gateways))
	for name := range pgf.gateways {
		gateways = append(gateways, name)
	}
	return gateways
}

// StripeGateway implements PaymentGateway for Stripe
type StripeGateway struct {
	apiKey string
}

// NewStripeGateway creates a new Stripe gateway instance
func NewStripeGateway() *StripeGateway {
	cfg := config.GetConfigManager()
	return &StripeGateway{
		apiKey: cfg.GetStripeConfig().APIKey,
	}
}

func (sg *StripeGateway) ProcessPayment(ctx context.Context, request *models.PaymentRequest) (*models.PaymentResponse, error) {
	log := logger.GetLogger()
	log.Info("Processing payment with Stripe", "amount", request.Amount, "currency", request.Currency)

	// Simulate Stripe payment processing
	response := &models.PaymentResponse{
		TransactionID: fmt.Sprintf("stripe_%s", request.ID),
		Status:        "success",
		Amount:        request.Amount,
		Currency:      request.Currency,
		Gateway:       "stripe",
		GatewayData: map[string]interface{}{
			"stripe_charge_id": fmt.Sprintf("ch_%s", request.ID),
			"payment_method":   request.PaymentMethod,
		},
	}

	return response, nil
}

func (sg *StripeGateway) RefundPayment(ctx context.Context, request *models.RefundRequest) (*models.RefundResponse, error) {
	log := logger.GetLogger()
	log.Info("Processing refund with Stripe", "payment_id", request.PaymentID, "amount", request.Amount)

	response := &models.RefundResponse{
		RefundID:      fmt.Sprintf("re_%s", request.PaymentID),
		Status:        "success",
		Amount:        request.Amount,
		PaymentID:     request.PaymentID,
		Gateway:       "stripe",
		GatewayData: map[string]interface{}{
			"stripe_refund_id": fmt.Sprintf("re_%s", request.PaymentID),
		},
	}

	return response, nil
}

func (sg *StripeGateway) GetPaymentStatus(ctx context.Context, paymentID string) (*models.PaymentStatus, error) {
	return &models.PaymentStatus{
		PaymentID: paymentID,
		Status:    "completed",
		Gateway:   "stripe",
	}, nil
}

func (sg *StripeGateway) ValidatePayment(request *models.PaymentRequest) error {
	if request.Amount <= 0 {
		return fmt.Errorf("invalid amount: %f", request.Amount)
	}
	if request.Currency == "" {
		return fmt.Errorf("currency is required")
	}
	if request.PaymentMethod == "" {
		return fmt.Errorf("payment method is required")
	}
	return nil
}

func (sg *StripeGateway) GetGatewayName() string {
	return "stripe"
}

// PayPalGateway implements PaymentGateway for PayPal
type PayPalGateway struct {
	clientID     string
	clientSecret string
}

// NewPayPalGateway creates a new PayPal gateway instance
func NewPayPalGateway() *PayPalGateway {
	cfg := config.GetConfigManager()
	paypalConfig := cfg.GetPayPalConfig()
	return &PayPalGateway{
		clientID:     paypalConfig.ClientID,
		clientSecret: paypalConfig.ClientSecret,
	}
}

func (ppg *PayPalGateway) ProcessPayment(ctx context.Context, request *models.PaymentRequest) (*models.PaymentResponse, error) {
	log := logger.GetLogger()
	log.Info("Processing payment with PayPal", "amount", request.Amount, "currency", request.Currency)

	response := &models.PaymentResponse{
		TransactionID: fmt.Sprintf("paypal_%s", request.ID),
		Status:        "success",
		Amount:        request.Amount,
		Currency:      request.Currency,
		Gateway:       "paypal",
		GatewayData: map[string]interface{}{
			"paypal_order_id": fmt.Sprintf("PAYPAL_%s", request.ID),
			"payment_method":  request.PaymentMethod,
		},
	}

	return response, nil
}

func (ppg *PayPalGateway) RefundPayment(ctx context.Context, request *models.RefundRequest) (*models.RefundResponse, error) {
	log := logger.GetLogger()
	log.Info("Processing refund with PayPal", "payment_id", request.PaymentID, "amount", request.Amount)

	response := &models.RefundResponse{
		RefundID:      fmt.Sprintf("paypal_refund_%s", request.PaymentID),
		Status:        "success",
		Amount:        request.Amount,
		PaymentID:     request.PaymentID,
		Gateway:       "paypal",
		GatewayData: map[string]interface{}{
			"paypal_refund_id": fmt.Sprintf("PAYPAL_REFUND_%s", request.PaymentID),
		},
	}

	return response, nil
}

func (ppg *PayPalGateway) GetPaymentStatus(ctx context.Context, paymentID string) (*models.PaymentStatus, error) {
	return &models.PaymentStatus{
		PaymentID: paymentID,
		Status:    "completed",
		Gateway:   "paypal",
	}, nil
}

func (ppg *PayPalGateway) ValidatePayment(request *models.PaymentRequest) error {
	if request.Amount <= 0 {
		return fmt.Errorf("invalid amount: %f", request.Amount)
	}
	if request.Currency == "" {
		return fmt.Errorf("currency is required")
	}
	return nil
}

func (ppg *PayPalGateway) GetGatewayName() string {
	return "paypal"
}

// RazorpayGateway implements PaymentGateway for Razorpay
type RazorpayGateway struct {
	keyID     string
	keySecret string
}

// NewRazorpayGateway creates a new Razorpay gateway instance
func NewRazorpayGateway() *RazorpayGateway {
	cfg := config.GetConfigManager()
	razorpayConfig := cfg.GetRazorpayConfig()
	return &RazorpayGateway{
		keyID:     razorpayConfig.KeyID,
		keySecret: razorpayConfig.KeySecret,
	}
}

func (rg *RazorpayGateway) ProcessPayment(ctx context.Context, request *models.PaymentRequest) (*models.PaymentResponse, error) {
	log := logger.GetLogger()
	log.Info("Processing payment with Razorpay", "amount", request.Amount, "currency", request.Currency)

	response := &models.PaymentResponse{
		TransactionID: fmt.Sprintf("razorpay_%s", request.ID),
		Status:        "success",
		Amount:        request.Amount,
		Currency:      request.Currency,
		Gateway:       "razorpay",
		GatewayData: map[string]interface{}{
			"razorpay_payment_id": fmt.Sprintf("pay_%s", request.ID),
			"order_id":            fmt.Sprintf("order_%s", request.ID),
		},
	}

	return response, nil
}

func (rg *RazorpayGateway) RefundPayment(ctx context.Context, request *models.RefundRequest) (*models.RefundResponse, error) {
	log := logger.GetLogger()
	log.Info("Processing refund with Razorpay", "payment_id", request.PaymentID, "amount", request.Amount)

	response := &models.RefundResponse{
		RefundID:      fmt.Sprintf("razorpay_refund_%s", request.PaymentID),
		Status:        "success",
		Amount:        request.Amount,
		PaymentID:     request.PaymentID,
		Gateway:       "razorpay",
		GatewayData: map[string]interface{}{
			"razorpay_refund_id": fmt.Sprintf("rfnd_%s", request.PaymentID),
		},
	}

	return response, nil
}

func (rg *RazorpayGateway) GetPaymentStatus(ctx context.Context, paymentID string) (*models.PaymentStatus, error) {
	return &models.PaymentStatus{
		PaymentID: paymentID,
		Status:    "completed",
		Gateway:   "razorpay",
	}, nil
}

func (rg *RazorpayGateway) ValidatePayment(request *models.PaymentRequest) error {
	if request.Amount <= 0 {
		return fmt.Errorf("invalid amount: %f", request.Amount)
	}
	if request.Currency == "" {
		return fmt.Errorf("currency is required")
	}
	return nil
}

func (rg *RazorpayGateway) GetGatewayName() string {
	return "razorpay"
}

// BankTransferGateway implements PaymentGateway for Bank Transfer
type BankTransferGateway struct {
	bankAPIKey string
}

// NewBankTransferGateway creates a new Bank Transfer gateway instance
func NewBankTransferGateway() *BankTransferGateway {
	cfg := config.GetConfigManager()
	return &BankTransferGateway{
		bankAPIKey: cfg.GetBankTransferConfig().APIKey,
	}
}

func (btg *BankTransferGateway) ProcessPayment(ctx context.Context, request *models.PaymentRequest) (*models.PaymentResponse, error) {
	log := logger.GetLogger()
	log.Info("Processing payment with Bank Transfer", "amount", request.Amount, "currency", request.Currency)

	response := &models.PaymentResponse{
		TransactionID: fmt.Sprintf("bank_%s", request.ID),
		Status:        "pending",
		Amount:        request.Amount,
		Currency:      request.Currency,
		Gateway:       "bank_transfer",
		GatewayData: map[string]interface{}{
			"bank_reference": fmt.Sprintf("BANK_%s", request.ID),
			"account_number": request.BankDetails.AccountNumber,
		},
	}

	return response, nil
}

func (btg *BankTransferGateway) RefundPayment(ctx context.Context, request *models.RefundRequest) (*models.RefundResponse, error) {
	log := logger.GetLogger()
	log.Info("Processing refund with Bank Transfer", "payment_id", request.PaymentID, "amount", request.Amount)

	response := &models.RefundResponse{
		RefundID:      fmt.Sprintf("bank_refund_%s", request.PaymentID),
		Status:        "pending",
		Amount:        request.Amount,
		PaymentID:     request.PaymentID,
		Gateway:       "bank_transfer",
		GatewayData: map[string]interface{}{
			"bank_refund_reference": fmt.Sprintf("BANK_REFUND_%s", request.PaymentID),
		},
	}

	return response, nil
}

func (btg *BankTransferGateway) GetPaymentStatus(ctx context.Context, paymentID string) (*models.PaymentStatus, error) {
	return &models.PaymentStatus{
		PaymentID: paymentID,
		Status:    "pending",
		Gateway:   "bank_transfer",
	}, nil
}

func (btg *BankTransferGateway) ValidatePayment(request *models.PaymentRequest) error {
	if request.Amount <= 0 {
		return fmt.Errorf("invalid amount: %f", request.Amount)
	}
	if request.Currency == "" {
		return fmt.Errorf("currency is required")
	}
	if request.BankDetails.AccountNumber == "" {
		return fmt.Errorf("bank account number is required")
	}
	return nil
}

func (btg *BankTransferGateway) GetGatewayName() string {
	return "bank_transfer"
}

// DigitalWalletGateway implements PaymentGateway for Digital Wallet
type DigitalWalletGateway struct {
	walletProvider string
	apiKey         string
}

// NewDigitalWalletGateway creates a new Digital Wallet gateway instance
func NewDigitalWalletGateway() *DigitalWalletGateway {
	cfg := config.GetConfigManager()
	walletConfig := cfg.GetDigitalWalletConfig()
	return &DigitalWalletGateway{
		walletProvider: walletConfig.Provider,
		apiKey:         walletConfig.APIKey,
	}
}

func (dwg *DigitalWalletGateway) ProcessPayment(ctx context.Context, request *models.PaymentRequest) (*models.PaymentResponse, error) {
	log := logger.GetLogger()
	log.Info("Processing payment with Digital Wallet", "amount", request.Amount, "currency", request.Currency)

	response := &models.PaymentResponse{
		TransactionID: fmt.Sprintf("wallet_%s", request.ID),
		Status:        "success",
		Amount:        request.Amount,
		Currency:      request.Currency,
		Gateway:       "digital_wallet",
		GatewayData: map[string]interface{}{
			"wallet_transaction_id": fmt.Sprintf("WALLET_%s", request.ID),
			"wallet_id":             request.WalletDetails.WalletID,
		},
	}

	return response, nil
}

func (dwg *DigitalWalletGateway) RefundPayment(ctx context.Context, request *models.RefundRequest) (*models.RefundResponse, error) {
	log := logger.GetLogger()
	log.Info("Processing refund with Digital Wallet", "payment_id", request.PaymentID, "amount", request.Amount)

	response := &models.RefundResponse{
		RefundID:      fmt.Sprintf("wallet_refund_%s", request.PaymentID),
		Status:        "success",
		Amount:        request.Amount,
		PaymentID:     request.PaymentID,
		Gateway:       "digital_wallet",
		GatewayData: map[string]interface{}{
			"wallet_refund_id": fmt.Sprintf("WALLET_REFUND_%s", request.PaymentID),
		},
	}

	return response, nil
}

func (dwg *DigitalWalletGateway) GetPaymentStatus(ctx context.Context, paymentID string) (*models.PaymentStatus, error) {
	return &models.PaymentStatus{
		PaymentID: paymentID,
		Status:    "completed",
		Gateway:   "digital_wallet",
	}, nil
}

func (dwg *DigitalWalletGateway) ValidatePayment(request *models.PaymentRequest) error {
	if request.Amount <= 0 {
		return fmt.Errorf("invalid amount: %f", request.Amount)
	}
	if request.Currency == "" {
		return fmt.Errorf("currency is required")
	}
	if request.WalletDetails.WalletID == "" {
		return fmt.Errorf("wallet ID is required")
	}
	return nil
}

func (dwg *DigitalWalletGateway) GetGatewayName() string {
	return "digital_wallet"
}
