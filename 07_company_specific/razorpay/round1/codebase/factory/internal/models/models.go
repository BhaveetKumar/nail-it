package models

import "time"

// PaymentRequest represents a payment request
type PaymentRequest struct {
	ID            string                 `json:"id"`
	UserID        string                 `json:"user_id"`
	Amount        float64                `json:"amount"`
	Currency      string                 `json:"currency"`
	PaymentMethod string                 `json:"payment_method"`
	BankDetails   BankDetails            `json:"bank_details,omitempty"`
	WalletDetails WalletDetails          `json:"wallet_details,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt     time.Time              `json:"created_at"`
}

// PaymentResponse represents a payment response
type PaymentResponse struct {
	TransactionID string                 `json:"transaction_id"`
	Status        string                 `json:"status"`
	Amount        float64                `json:"amount"`
	Currency      string                 `json:"currency"`
	Gateway       string                 `json:"gateway"`
	GatewayData   map[string]interface{} `json:"gateway_data,omitempty"`
	ProcessedAt   time.Time              `json:"processed_at"`
}

// RefundRequest represents a refund request
type RefundRequest struct {
	ID        string    `json:"id"`
	PaymentID string    `json:"payment_id"`
	Amount    float64   `json:"amount"`
	Reason    string    `json:"reason,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

// RefundResponse represents a refund response
type RefundResponse struct {
	RefundID    string                 `json:"refund_id"`
	Status      string                 `json:"status"`
	Amount      float64                `json:"amount"`
	PaymentID   string                 `json:"payment_id"`
	Gateway     string                 `json:"gateway"`
	GatewayData map[string]interface{} `json:"gateway_data,omitempty"`
	ProcessedAt time.Time              `json:"processed_at"`
}

// PaymentStatus represents payment status information
type PaymentStatus struct {
	PaymentID string    `json:"payment_id"`
	Status    string    `json:"status"`
	Gateway   string    `json:"gateway"`
	UpdatedAt time.Time `json:"updated_at"`
}

// BankDetails represents bank transfer details
type BankDetails struct {
	AccountNumber string `json:"account_number"`
	RoutingNumber string `json:"routing_number"`
	BankName      string `json:"bank_name"`
	AccountHolder string `json:"account_holder"`
}

// WalletDetails represents digital wallet details
type WalletDetails struct {
	WalletID   string `json:"wallet_id"`
	WalletType string `json:"wallet_type"`
	Provider   string `json:"provider"`
}

// NotificationRequest represents a notification request
type NotificationRequest struct {
	ID        string                 `json:"id"`
	Recipient string                 `json:"recipient"`
	Subject   string                 `json:"subject,omitempty"`
	Title     string                 `json:"title,omitempty"`
	Message   string                 `json:"message"`
	Channel   string                 `json:"channel,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
}

// NotificationResponse represents a notification response
type NotificationResponse struct {
	MessageID   string                 `json:"message_id"`
	Status      string                 `json:"status"`
	Channel     string                 `json:"channel"`
	Recipient   string                 `json:"recipient"`
	ChannelData map[string]interface{} `json:"channel_data,omitempty"`
	SentAt      time.Time              `json:"sent_at"`
}

// User represents a user in the system
type User struct {
	ID        string    `json:"id"`
	Email     string    `json:"email"`
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// CreateUserRequest represents the request to create a user
type CreateUserRequest struct {
	Email string `json:"email" binding:"required,email"`
	Name  string `json:"name" binding:"required"`
}

// UpdateUserRequest represents the request to update a user
type UpdateUserRequest struct {
	Name string `json:"name" binding:"required"`
}

// CreatePaymentRequest represents the request to create a payment
type CreatePaymentRequest struct {
	UserID        string         `json:"user_id" binding:"required"`
	Amount        float64        `json:"amount" binding:"required,gt=0"`
	Currency      string         `json:"currency" binding:"required,len=3"`
	PaymentMethod string         `json:"payment_method" binding:"required"`
	BankDetails   *BankDetails   `json:"bank_details,omitempty"`
	WalletDetails *WalletDetails `json:"wallet_details,omitempty"`
}

// UpdatePaymentStatusRequest represents the request to update payment status
type UpdatePaymentStatusRequest struct {
	Status string `json:"status" binding:"required,oneof=pending processing completed failed"`
}

// SendNotificationRequest represents the request to send a notification
type SendNotificationRequest struct {
	Recipient string                 `json:"recipient" binding:"required"`
	Subject   string                 `json:"subject,omitempty"`
	Title     string                 `json:"title,omitempty"`
	Message   string                 `json:"message" binding:"required"`
	Channel   string                 `json:"channel,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// WebSocketMessage represents a WebSocket message
type WebSocketMessage struct {
	Type      string      `json:"type"`
	Data      interface{} `json:"data"`
	Timestamp int64       `json:"timestamp"`
	UserID    string      `json:"user_id,omitempty"`
	ClientID  string      `json:"client_id,omitempty"`
}

// KafkaEvent represents a Kafka event
type KafkaEvent struct {
	Type      string      `json:"type"`
	Data      interface{} `json:"data"`
	Timestamp int64       `json:"timestamp"`
	Source    string      `json:"source"`
}

// PaymentEvent represents a payment-related event
type PaymentEvent struct {
	Type      string      `json:"type"`
	PaymentID string      `json:"payment_id"`
	Data      interface{} `json:"data"`
	Timestamp int64       `json:"timestamp"`
	Source    string      `json:"source"`
}

// NotificationEvent represents a notification-related event
type NotificationEvent struct {
	Type      string      `json:"type"`
	Data      interface{} `json:"data"`
	Timestamp int64       `json:"timestamp"`
	Source    string      `json:"source"`
}

// HealthCheckResponse represents the health check response
type HealthCheckResponse struct {
	Status           string `json:"status"`
	ConnectedClients int    `json:"connected_clients"`
	ConnectedUsers   int    `json:"connected_users"`
	Timestamp        int64  `json:"timestamp"`
	Error            string `json:"error,omitempty"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error     string `json:"error"`
	Message   string `json:"message,omitempty"`
	Timestamp int64  `json:"timestamp"`
}

// SuccessResponse represents a success response
type SuccessResponse struct {
	Message   string      `json:"message"`
	Data      interface{} `json:"data,omitempty"`
	Timestamp int64       `json:"timestamp"`
}

// PaginationResponse represents a paginated response
type PaginationResponse struct {
	Data       interface{} `json:"data"`
	Page       int         `json:"page"`
	Limit      int         `json:"limit"`
	Count      int         `json:"count"`
	TotalPages int         `json:"total_pages"`
	HasNext    bool        `json:"has_next"`
	HasPrev    bool        `json:"has_prev"`
}

// FactoryInfo represents information about a factory
type FactoryInfo struct {
	FactoryType string   `json:"factory_type"`
	Available   []string `json:"available"`
	Active      string   `json:"active,omitempty"`
}

// SystemInfo represents information about the system
type SystemInfo struct {
	PaymentGateway      string `json:"payment_gateway"`
	NotificationChannel string `json:"notification_channel"`
	DatabaseType        string `json:"database_type"`
}

// MultiChannelNotificationResponse represents a multi-channel notification response
type MultiChannelNotificationResponse struct {
	Responses []*NotificationResponse `json:"responses"`
	Success   int                     `json:"success"`
	Failed    int                     `json:"failed"`
	Total     int                     `json:"total"`
}

// DatabaseQueryResponse represents a database query response
type DatabaseQueryResponse struct {
	Results map[string]interface{} `json:"results"`
	Success int                    `json:"success"`
	Failed  int                    `json:"failed"`
	Total   int                    `json:"total"`
}
