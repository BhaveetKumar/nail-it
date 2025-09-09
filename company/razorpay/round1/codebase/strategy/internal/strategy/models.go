package strategy

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
	Gateway       string            `json:"gateway" validate:"required"`
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
	Gateway       string            `json:"gateway"`
	ProcessedAt   time.Time         `json:"processed_at"`
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
	SentAt         time.Time         `json:"sent_at"`
	DeliveryID     string            `json:"delivery_id"`
	Metadata       map[string]string `json:"metadata"`
	Error          string            `json:"error,omitempty"`
}

// PricingRequest represents a pricing request
type PricingRequest struct {
	PricingID    string            `json:"pricing_id" validate:"required"`
	ProductID    string            `json:"product_id" validate:"required"`
	UserID       string            `json:"user_id" validate:"required"`
	Quantity     int               `json:"quantity" validate:"required,gt=0"`
	BasePrice    float64           `json:"base_price" validate:"required,gt=0"`
	Currency     string            `json:"currency" validate:"required,len=3"`
	DiscountCode string            `json:"discount_code"`
	Metadata     map[string]string `json:"metadata"`
	CreatedAt    time.Time         `json:"created_at"`
	UpdatedAt    time.Time         `json:"updated_at"`
}

// PricingResponse represents a pricing response
type PricingResponse struct {
	PricingID     string            `json:"pricing_id"`
	ProductID     string            `json:"product_id"`
	BasePrice     float64           `json:"base_price"`
	DiscountPrice float64           `json:"discount_price"`
	FinalPrice    float64           `json:"final_price"`
	Currency      string            `json:"currency"`
	DiscountCode  string            `json:"discount_code"`
	CalculatedAt  time.Time         `json:"calculated_at"`
	Metadata      map[string]string `json:"metadata"`
	Error         string            `json:"error,omitempty"`
}

// AuthRequest represents an authentication request
type AuthRequest struct {
	AuthID       string            `json:"auth_id" validate:"required"`
	UserID       string            `json:"user_id" validate:"required"`
	Method       string            `json:"method" validate:"required"`
	Credentials  map[string]string `json:"credentials" validate:"required"`
	IPAddress    string            `json:"ip_address"`
	UserAgent    string            `json:"user_agent"`
	Metadata     map[string]string `json:"metadata"`
	CreatedAt    time.Time         `json:"created_at"`
	UpdatedAt    time.Time         `json:"updated_at"`
}

// AuthResponse represents an authentication response
type AuthResponse struct {
	AuthID      string            `json:"auth_id"`
	UserID      string            `json:"user_id"`
	Status      string            `json:"status"`
	Token       string            `json:"token"`
	ExpiresAt   time.Time         `json:"expires_at"`
	Method      string            `json:"method"`
	AuthenticatedAt time.Time      `json:"authenticated_at"`
	Metadata    map[string]string `json:"metadata"`
	Error       string            `json:"error,omitempty"`
}

// CacheRequest represents a cache request
type CacheRequest struct {
	Key        string        `json:"key" validate:"required"`
	Value      interface{}   `json:"value"`
	TTL        time.Duration `json:"ttl"`
	Operation  string        `json:"operation" validate:"required"`
	CreatedAt  time.Time     `json:"created_at"`
	UpdatedAt  time.Time     `json:"updated_at"`
}

// CacheResponse represents a cache response
type CacheResponse struct {
	Key       string      `json:"key"`
	Value     interface{} `json:"value"`
	Hit       bool        `json:"hit"`
	TTL       time.Duration `json:"ttl"`
	ExpiresAt time.Time   `json:"expires_at"`
	Error     string      `json:"error,omitempty"`
}

// LogRequest represents a log request
type LogRequest struct {
	LogID     string            `json:"log_id" validate:"required"`
	Level     LogLevel          `json:"level" validate:"required"`
	Message   string            `json:"message" validate:"required"`
	Fields    map[string]interface{} `json:"fields"`
	Timestamp time.Time         `json:"timestamp"`
	Service   string            `json:"service"`
	Version   string            `json:"version"`
}

// LogResponse represents a log response
type LogResponse struct {
	LogID     string    `json:"log_id"`
	Status    string    `json:"status"`
	LoggedAt  time.Time `json:"logged_at"`
	Error     string    `json:"error,omitempty"`
}

// DataProcessingRequest represents a data processing request
type DataProcessingRequest struct {
	RequestID   string      `json:"request_id" validate:"required"`
	Data        interface{} `json:"data" validate:"required"`
	Format      string      `json:"format" validate:"required"`
	Operation   string      `json:"operation" validate:"required"`
	Parameters  map[string]string `json:"parameters"`
	CreatedAt   time.Time   `json:"created_at"`
	UpdatedAt   time.Time   `json:"updated_at"`
}

// DataProcessingResponse represents a data processing response
type DataProcessingResponse struct {
	RequestID    string      `json:"request_id"`
	ProcessedData interface{} `json:"processed_data"`
	Format       string      `json:"format"`
	Operation    string      `json:"operation"`
	ProcessedAt  time.Time   `json:"processed_at"`
	Metadata     map[string]string `json:"metadata"`
	Error        string      `json:"error,omitempty"`
}

// StrategyRequest represents a generic strategy request
type StrategyRequest struct {
	RequestID   string            `json:"request_id" validate:"required"`
	StrategyType string           `json:"strategy_type" validate:"required"`
	StrategyName string           `json:"strategy_name" validate:"required"`
	Data        interface{}       `json:"data" validate:"required"`
	Parameters  map[string]string `json:"parameters"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// StrategyResponse represents a generic strategy response
type StrategyResponse struct {
	RequestID    string            `json:"request_id"`
	StrategyType string            `json:"strategy_type"`
	StrategyName string            `json:"strategy_name"`
	Result       interface{}       `json:"result"`
	ProcessedAt  time.Time         `json:"processed_at"`
	Metadata     map[string]string `json:"metadata"`
	Error        string            `json:"error,omitempty"`
}

// StrategyInfo represents information about a strategy
type StrategyInfo struct {
	Name                string            `json:"name"`
	Type                string            `json:"type"`
	Description         string            `json:"description"`
	Version             string            `json:"version"`
	Author              string            `json:"author"`
	SupportedFeatures   []string          `json:"supported_features"`
	Configuration       map[string]string `json:"configuration"`
	IsAvailable         bool              `json:"is_available"`
	LastHealthCheck     time.Time         `json:"last_health_check"`
	CreatedAt           time.Time         `json:"created_at"`
	UpdatedAt           time.Time         `json:"updated_at"`
}

// StrategyHealth represents the health status of a strategy
type StrategyHealth struct {
	StrategyName string            `json:"strategy_name"`
	Status       string            `json:"status"`
	Message      string            `json:"message"`
	LastCheck    time.Time         `json:"last_check"`
	Metrics      map[string]interface{} `json:"metrics"`
	Error        string            `json:"error,omitempty"`
}

// StrategySelection represents strategy selection criteria
type StrategySelection struct {
	StrategyType string            `json:"strategy_type"`
	Criteria     map[string]string `json:"criteria"`
	Priority     int               `json:"priority"`
	Fallback     string            `json:"fallback"`
	Timeout      time.Duration     `json:"timeout"`
	RetryCount   int               `json:"retry_count"`
}

// StrategyExecution represents strategy execution details
type StrategyExecution struct {
	ExecutionID  string            `json:"execution_id"`
	StrategyName string            `json:"strategy_name"`
	StrategyType string            `json:"strategy_type"`
	Status       string            `json:"status"`
	StartTime    time.Time         `json:"start_time"`
	EndTime      time.Time         `json:"end_time"`
	Duration     time.Duration     `json:"duration"`
	Input        interface{}       `json:"input"`
	Output       interface{}       `json:"output"`
	Error        string            `json:"error,omitempty"`
	Metadata     map[string]string `json:"metadata"`
}
