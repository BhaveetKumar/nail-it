package bridge

import (
	"time"
)

// PaymentRequest represents a payment request
type PaymentRequest struct {
	ID          string                 `json:"id"`
	Amount      float64                `json:"amount"`
	Currency    string                 `json:"currency"`
	CustomerID  string                 `json:"customer_id"`
	MerchantID  string                 `json:"merchant_id"`
	Description string                 `json:"description"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
}

// PaymentResponse represents a payment response
type PaymentResponse struct {
	ID            string                 `json:"id"`
	Status        string                 `json:"status"`
	TransactionID string                 `json:"transaction_id"`
	Amount        float64                `json:"amount"`
	Currency      string                 `json:"currency"`
	Gateway       string                 `json:"gateway"`
	Metadata      map[string]interface{} `json:"metadata"`
	CreatedAt     time.Time              `json:"created_at"`
	ProcessedAt   time.Time              `json:"processed_at"`
}

// NotificationRequest represents a notification request
type NotificationRequest struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"`
	Recipient string                 `json:"recipient"`
	Subject   string                 `json:"subject"`
	Message   string                 `json:"message"`
	Metadata  map[string]interface{} `json:"metadata"`
	CreatedAt time.Time              `json:"created_at"`
}

// NotificationResponse represents a notification response
type NotificationResponse struct {
	ID        string                 `json:"id"`
	Status    string                 `json:"status"`
	Channel   string                 `json:"channel"`
	MessageID string                 `json:"message_id"`
	Metadata  map[string]interface{} `json:"metadata"`
	CreatedAt time.Time              `json:"created_at"`
	SentAt    time.Time              `json:"sent_at"`
}

// PaymentGatewayConfig represents payment gateway configuration
type PaymentGatewayConfig struct {
	Name        string            `json:"name"`
	APIKey      string            `json:"api_key"`
	SecretKey   string            `json:"secret_key"`
	BaseURL     string            `json:"base_url"`
	Environment string            `json:"environment"`
	Settings    map[string]string `json:"settings"`
}

// NotificationChannelConfig represents notification channel configuration
type NotificationChannelConfig struct {
	Name        string            `json:"name"`
	APIKey      string            `json:"api_key"`
	SecretKey   string            `json:"secret_key"`
	BaseURL     string            `json:"base_url"`
	Environment string            `json:"environment"`
	Settings    map[string]string `json:"settings"`
}

// PaymentMetrics represents payment processing metrics
type PaymentMetrics struct {
	TotalProcessed     int64     `json:"total_processed"`
	SuccessfulPayments int64     `json:"successful_payments"`
	FailedPayments     int64     `json:"failed_payments"`
	TotalAmount        float64   `json:"total_amount"`
	AverageAmount      float64   `json:"average_amount"`
	SuccessRate        float64   `json:"success_rate"`
	LastProcessed      time.Time `json:"last_processed"`
}

// NotificationMetrics represents notification sending metrics
type NotificationMetrics struct {
	TotalSent       int64     `json:"total_sent"`
	SuccessfulSends int64     `json:"successful_sends"`
	FailedSends     int64     `json:"failed_sends"`
	SuccessRate     float64   `json:"success_rate"`
	LastSent        time.Time `json:"last_sent"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error     string    `json:"error"`
	Code      string    `json:"code"`
	Message   string    `json:"message"`
	Timestamp time.Time `json:"timestamp"`
}
