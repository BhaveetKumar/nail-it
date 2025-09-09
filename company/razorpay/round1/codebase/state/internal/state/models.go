package state

import (
	"time"
)

// PaymentState represents payment states
type PaymentState struct {
	PaymentID     string            `json:"payment_id" validate:"required"`
	UserID        string            `json:"user_id" validate:"required"`
	Amount        float64           `json:"amount" validate:"required,gt=0"`
	Currency      string            `json:"currency" validate:"required,len=3"`
	PaymentMethod string            `json:"payment_method" validate:"required"`
	Gateway       string            `json:"gateway" validate:"required"`
	Description   string            `json:"description"`
	Status        string            `json:"status" validate:"required"`
	Metadata      map[string]string `json:"metadata"`
	CreatedAt     time.Time         `json:"created_at"`
	UpdatedAt     time.Time         `json:"updated_at"`
}

// PaymentStateResult represents the result of payment state operations
type PaymentStateResult struct {
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

// OrderState represents order states
type OrderState struct {
	OrderID       string            `json:"order_id" validate:"required"`
	UserID        string            `json:"user_id" validate:"required"`
	Status        string            `json:"status" validate:"required"`
	Items         []OrderItem       `json:"items" validate:"required"`
	TotalAmount   float64           `json:"total_amount" validate:"required,gt=0"`
	Currency      string            `json:"currency" validate:"required,len=3"`
	ShippingAddress string          `json:"shipping_address"`
	BillingAddress  string          `json:"billing_address"`
	Metadata      map[string]string `json:"metadata"`
	CreatedAt     time.Time         `json:"created_at"`
	UpdatedAt     time.Time         `json:"updated_at"`
}

// OrderItem represents an order item
type OrderItem struct {
	ProductID string  `json:"product_id" validate:"required"`
	Quantity  int     `json:"quantity" validate:"required,gt=0"`
	Price     float64 `json:"price" validate:"required,gt=0"`
	Name      string  `json:"name" validate:"required"`
}

// OrderStateResult represents the result of order state operations
type OrderStateResult struct {
	OrderID       string            `json:"order_id"`
	Status        string            `json:"status"`
	Items         []OrderItem       `json:"items"`
	TotalAmount   float64           `json:"total_amount"`
	Currency      string            `json:"currency"`
	ProcessedAt   time.Time         `json:"processed_at"`
	Metadata      map[string]string `json:"metadata"`
	Error         string            `json:"error,omitempty"`
}

// UserState represents user states
type UserState struct {
	UserID      string            `json:"user_id" validate:"required"`
	Status      string            `json:"status" validate:"required"`
	Email       string            `json:"email" validate:"required,email"`
	Name        string            `json:"name" validate:"required"`
	Phone       string            `json:"phone"`
	Address     string            `json:"address"`
	Metadata    map[string]string `json:"metadata"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// UserStateResult represents the result of user state operations
type UserStateResult struct {
	UserID      string            `json:"user_id"`
	Status      string            `json:"status"`
	Email       string            `json:"email"`
	Name        string            `json:"name"`
	Phone       string            `json:"phone"`
	Address     string            `json:"address"`
	ProcessedAt time.Time         `json:"processed_at"`
	Metadata    map[string]string `json:"metadata"`
	Error       string            `json:"error,omitempty"`
}

// InventoryState represents inventory states
type InventoryState struct {
	ProductID   string            `json:"product_id" validate:"required"`
	Status      string            `json:"status" validate:"required"`
	Name        string            `json:"name" validate:"required"`
	Description string            `json:"description"`
	Price       float64           `json:"price" validate:"required,gt=0"`
	Stock       int               `json:"stock" validate:"required,gte=0"`
	Category    string            `json:"category"`
	Metadata    map[string]string `json:"metadata"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// InventoryStateResult represents the result of inventory state operations
type InventoryStateResult struct {
	ProductID   string            `json:"product_id"`
	Status      string            `json:"status"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Price       float64           `json:"price"`
	Stock       int               `json:"stock"`
	Category    string            `json:"category"`
	ProcessedAt time.Time         `json:"processed_at"`
	Metadata    map[string]string `json:"metadata"`
	Error       string            `json:"error,omitempty"`
}

// NotificationState represents notification states
type NotificationState struct {
	NotificationID string            `json:"notification_id" validate:"required"`
	UserID         string            `json:"user_id" validate:"required"`
	Status         string            `json:"status" validate:"required"`
	Channel        string            `json:"channel" validate:"required"`
	Type           string            `json:"type" validate:"required"`
	Title          string            `json:"title" validate:"required"`
	Message        string            `json:"message" validate:"required"`
	Priority       string            `json:"priority"`
	Metadata       map[string]string `json:"metadata"`
	CreatedAt      time.Time         `json:"created_at"`
	UpdatedAt      time.Time         `json:"updated_at"`
}

// NotificationStateResult represents the result of notification state operations
type NotificationStateResult struct {
	NotificationID string            `json:"notification_id"`
	Status         string            `json:"status"`
	Channel        string            `json:"channel"`
	Type           string            `json:"type"`
	Title          string            `json:"title"`
	Message        string            `json:"message"`
	Priority       string            `json:"priority"`
	SentAt         time.Time         `json:"sent_at"`
	Metadata       map[string]string `json:"metadata"`
	Error          string            `json:"error,omitempty"`
}

// RefundState represents refund states
type RefundState struct {
	RefundID   string            `json:"refund_id" validate:"required"`
	PaymentID  string            `json:"payment_id" validate:"required"`
	UserID     string            `json:"user_id" validate:"required"`
	Status     string            `json:"status" validate:"required"`
	Amount     float64           `json:"amount" validate:"required,gt=0"`
	Currency   string            `json:"currency" validate:"required,len=3"`
	Reason     string            `json:"reason" validate:"required"`
	Metadata   map[string]string `json:"metadata"`
	CreatedAt  time.Time         `json:"created_at"`
	UpdatedAt  time.Time         `json:"updated_at"`
}

// RefundStateResult represents the result of refund state operations
type RefundStateResult struct {
	RefundID   string            `json:"refund_id"`
	Status     string            `json:"status"`
	PaymentID  string            `json:"payment_id"`
	UserID     string            `json:"user_id"`
	Amount     float64           `json:"amount"`
	Currency   string            `json:"currency"`
	Reason     string            `json:"reason"`
	ProcessedAt time.Time        `json:"processed_at"`
	Metadata   map[string]string `json:"metadata"`
	Error      string            `json:"error,omitempty"`
}

// AuditState represents audit states
type AuditState struct {
	AuditID    string            `json:"audit_id" validate:"required"`
	EntityType string            `json:"entity_type" validate:"required"`
	EntityID   string            `json:"entity_id" validate:"required"`
	Status     string            `json:"status" validate:"required"`
	Action     string            `json:"action" validate:"required"`
	Changes    map[string]interface{} `json:"changes" validate:"required"`
	UserID     string            `json:"user_id"`
	IPAddress  string            `json:"ip_address"`
	UserAgent  string            `json:"user_agent"`
	Metadata   map[string]string `json:"metadata"`
	CreatedAt  time.Time         `json:"created_at"`
	UpdatedAt  time.Time         `json:"updated_at"`
}

// AuditStateResult represents the result of audit state operations
type AuditStateResult struct {
	AuditID    string            `json:"audit_id"`
	Status     string            `json:"status"`
	EntityType string            `json:"entity_type"`
	EntityID   string            `json:"entity_id"`
	Action     string            `json:"action"`
	Changes    map[string]interface{} `json:"changes"`
	UserID     string            `json:"user_id"`
	IPAddress  string            `json:"ip_address"`
	UserAgent  string            `json:"user_agent"`
	ProcessedAt time.Time        `json:"processed_at"`
	Metadata   map[string]string `json:"metadata"`
	Error      string            `json:"error,omitempty"`
}

// SystemState represents system states
type SystemState struct {
	SystemID   string            `json:"system_id" validate:"required"`
	Status     string            `json:"status" validate:"required"`
	Component  string            `json:"component" validate:"required"`
	Version    string            `json:"version"`
	Health     string            `json:"health"`
	Metadata   map[string]string `json:"metadata"`
	CreatedAt  time.Time         `json:"created_at"`
	UpdatedAt  time.Time         `json:"updated_at"`
}

// SystemStateResult represents the result of system state operations
type SystemStateResult struct {
	SystemID   string            `json:"system_id"`
	Status     string            `json:"status"`
	Component  string            `json:"component"`
	Version    string            `json:"version"`
	Health     string            `json:"health"`
	ProcessedAt time.Time        `json:"processed_at"`
	Metadata   map[string]string `json:"metadata"`
	Error      string            `json:"error,omitempty"`
}

// StateRequest represents a generic state request
type StateRequest struct {
	EntityID   string            `json:"entity_id" validate:"required"`
	EntityType string            `json:"entity_type" validate:"required"`
	State      string            `json:"state" validate:"required"`
	Data       interface{}       `json:"data" validate:"required"`
	Metadata   map[string]string `json:"metadata"`
	CreatedAt  time.Time         `json:"created_at"`
	UpdatedAt  time.Time         `json:"updated_at"`
}

// StateResponse represents a generic state response
type StateResponse struct {
	EntityID   string            `json:"entity_id"`
	EntityType string            `json:"entity_type"`
	State      string            `json:"state"`
	Result     interface{}       `json:"result"`
	ProcessedAt time.Time        `json:"processed_at"`
	Metadata   map[string]string `json:"metadata"`
	Error      string            `json:"error,omitempty"`
}

// StateInfo represents information about a state
type StateInfo struct {
	StateName           string            `json:"state_name"`
	StateType           string            `json:"state_type"`
	Description         string            `json:"description"`
	IsFinal             bool              `json:"is_final"`
	AllowedTransitions  []string          `json:"allowed_transitions"`
	Metadata            map[string]string `json:"metadata"`
	CreatedAt           time.Time         `json:"created_at"`
	UpdatedAt           time.Time         `json:"updated_at"`
}

// StateHealth represents the health status of a state
type StateHealth struct {
	StateName string            `json:"state_name"`
	Status    string            `json:"status"`
	Message   string            `json:"message"`
	LastCheck time.Time         `json:"last_check"`
	Metrics   map[string]interface{} `json:"metrics"`
	Error     string            `json:"error,omitempty"`
}

// StateTrend represents a trend data point for states
type StateTrend struct {
	Timestamp   time.Time `json:"timestamp"`
	StateName   string    `json:"state_name"`
	EntityType  string    `json:"entity_type"`
	Count       int       `json:"count"`
	SuccessRate float64   `json:"success_rate"`
	AvgDuration time.Duration `json:"avg_duration"`
}
