package adapter

import (
	"time"
)

// PaymentRequest represents a payment request
type PaymentRequest struct {
	PaymentID     string            `json:"payment_id" validate:"required"`
	UserID        string            `json:"user_id" validate:"required"`
	Amount        float64           `json:"amount" validate:"required,gt=0"`
	Currency      string            `json:"currency" validate:"required,len=3"`
	PaymentMethod string            `json:"payment_method" validate:"required"`
	Description   string            `json:"description"`
	Metadata      map[string]string `json:"metadata"`
	CreatedAt     time.Time         `json:"created_at"`
	UpdatedAt     time.Time         `json:"updated_at"`
}

// PaymentResponse represents a payment response
type PaymentResponse struct {
	PaymentID     string            `json:"payment_id"`
	Status        string            `json:"status"`
	TransactionID string            `json:"transaction_id"`
	Amount        float64           `json:"amount"`
	Currency      string            `json:"currency"`
	ProcessedAt   time.Time         `json:"processed_at"`
	Metadata      map[string]string `json:"metadata"`
	Error         string            `json:"error,omitempty"`
}

// RefundRequest represents a refund request
type RefundRequest struct {
	RefundID  string            `json:"refund_id" validate:"required"`
	PaymentID string            `json:"payment_id" validate:"required"`
	Amount    float64           `json:"amount" validate:"required,gt=0"`
	Currency  string            `json:"currency" validate:"required,len=3"`
	Reason    string            `json:"reason" validate:"required"`
	Metadata  map[string]string `json:"metadata"`
	CreatedAt time.Time         `json:"created_at"`
	UpdatedAt time.Time         `json:"updated_at"`
}

// RefundResponse represents a refund response
type RefundResponse struct {
	RefundID    string            `json:"refund_id"`
	PaymentID   string            `json:"payment_id"`
	Status      string            `json:"status"`
	Amount      float64           `json:"amount"`
	Currency    string            `json:"currency"`
	ProcessedAt time.Time         `json:"processed_at"`
	Metadata    map[string]string `json:"metadata"`
	Error       string            `json:"error,omitempty"`
}

// PaymentStatus represents payment status
type PaymentStatus struct {
	PaymentID     string            `json:"payment_id"`
	Status        string            `json:"status"`
	Amount        float64           `json:"amount"`
	Currency      string            `json:"currency"`
	LastUpdated   time.Time         `json:"last_updated"`
	Metadata      map[string]string `json:"metadata"`
	Error         string            `json:"error,omitempty"`
}

// NotificationRequest represents a notification request
type NotificationRequest struct {
	NotificationID string            `json:"notification_id" validate:"required"`
	UserID         string            `json:"user_id" validate:"required"`
	Channel        string            `json:"channel" validate:"required"`
	Type           string            `json:"type" validate:"required"`
	Title          string            `json:"title" validate:"required"`
	Message        string            `json:"message" validate:"required"`
	Priority       string            `json:"priority"`
	Metadata       map[string]string `json:"metadata"`
	CreatedAt      time.Time         `json:"created_at"`
	UpdatedAt      time.Time         `json:"updated_at"`
}

// NotificationResponse represents a notification response
type NotificationResponse struct {
	NotificationID string            `json:"notification_id"`
	Status         string            `json:"status"`
	Channel        string            `json:"channel"`
	Type           string            `json:"type"`
	SentAt         time.Time         `json:"sent_at"`
	DeliveryID     string            `json:"delivery_id"`
	Metadata       map[string]string `json:"metadata"`
	Error          string            `json:"error,omitempty"`
}

// NotificationStatus represents notification status
type NotificationStatus struct {
	NotificationID string            `json:"notification_id"`
	Status         string            `json:"status"`
	Channel        string            `json:"channel"`
	Type           string            `json:"type"`
	LastUpdated    time.Time         `json:"last_updated"`
	Metadata       map[string]string `json:"metadata"`
	Error          string            `json:"error,omitempty"`
}

// File represents a file
type File struct {
	ID          string            `json:"id" validate:"required"`
	Name        string            `json:"name" validate:"required"`
	Content     []byte            `json:"content" validate:"required"`
	ContentType string            `json:"content_type" validate:"required"`
	Size        int64             `json:"size" validate:"required,gt=0"`
	Metadata    map[string]string `json:"metadata"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// UploadResponse represents an upload response
type UploadResponse struct {
	FileID      string            `json:"file_id"`
	FileName    string            `json:"file_name"`
	FileSize    int64             `json:"file_size"`
	ContentType string            `json:"content_type"`
	UploadedAt  time.Time         `json:"uploaded_at"`
	URL         string            `json:"url"`
	Metadata    map[string]string `json:"metadata"`
	Error       string            `json:"error,omitempty"`
}

// FileInfo represents file information
type FileInfo struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Size        int64             `json:"size"`
	ContentType string            `json:"content_type"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	Metadata    map[string]string `json:"metadata"`
}

// Credentials represents authentication credentials
type Credentials struct {
	Username string            `json:"username" validate:"required"`
	Password string            `json:"password" validate:"required"`
	Metadata map[string]string `json:"metadata"`
}

// AuthResponse represents an authentication response
type AuthResponse struct {
	UserID      string            `json:"user_id"`
	Token       string            `json:"token"`
	RefreshToken string           `json:"refresh_token"`
	ExpiresAt   time.Time         `json:"expires_at"`
	UserInfo    map[string]interface{} `json:"user_info"`
	Metadata    map[string]string `json:"metadata"`
	Error       string            `json:"error,omitempty"`
}

// TokenValidation represents token validation result
type TokenValidation struct {
	Valid     bool              `json:"valid"`
	UserID    string            `json:"user_id"`
	ExpiresAt time.Time         `json:"expires_at"`
	UserInfo  map[string]interface{} `json:"user_info"`
	Metadata  map[string]string `json:"metadata"`
	Error     string            `json:"error,omitempty"`
}

// AdapterRequest represents a generic adapter request
type AdapterRequest struct {
	AdapterType string            `json:"adapter_type" validate:"required"`
	AdapterName string            `json:"adapter_name" validate:"required"`
	Operation   string            `json:"operation" validate:"required"`
	Data        interface{}       `json:"data" validate:"required"`
	Metadata    map[string]string `json:"metadata"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// AdapterResponse represents a generic adapter response
type AdapterResponse struct {
	AdapterType string            `json:"adapter_type"`
	AdapterName string            `json:"adapter_name"`
	Operation   string            `json:"operation"`
	Result      interface{}       `json:"result"`
	ProcessedAt time.Time         `json:"processed_at"`
	Metadata    map[string]string `json:"metadata"`
	Error       string            `json:"error,omitempty"`
}

// AdapterInfo represents information about an adapter
type AdapterInfo struct {
	AdapterType string            `json:"adapter_type"`
	AdapterName string            `json:"adapter_name"`
	Description string            `json:"description"`
	Version     string            `json:"version"`
	Status      string            `json:"status"`
	IsAvailable bool              `json:"is_available"`
	LastHealthCheck time.Time     `json:"last_health_check"`
	Metadata    map[string]string `json:"metadata"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// AdapterTrend represents a trend data point for adapters
type AdapterTrend struct {
	Timestamp   time.Time `json:"timestamp"`
	AdapterType string    `json:"adapter_type"`
	AdapterName string    `json:"adapter_name"`
	Count       int       `json:"count"`
	SuccessRate float64   `json:"success_rate"`
	AvgDuration time.Duration `json:"avg_duration"`
}
