package command

import (
	"time"
)

// PaymentCommand represents a payment command
type PaymentCommand struct {
	CommandID     string            `json:"command_id" validate:"required"`
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

// PaymentCommandResult represents the result of payment command execution
type PaymentCommandResult struct {
	CommandID     string            `json:"command_id"`
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

// UserCommand represents a user command
type UserCommand struct {
	CommandID string                 `json:"command_id" validate:"required"`
	UserID    string                 `json:"user_id" validate:"required"`
	Action    string                 `json:"action" validate:"required"`
	Data      map[string]interface{} `json:"data" validate:"required"`
	Metadata  map[string]string      `json:"metadata"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

// UserCommandResult represents the result of user command execution
type UserCommandResult struct {
	CommandID   string                 `json:"command_id"`
	UserID      string                 `json:"user_id"`
	Action      string                 `json:"action"`
	Status      string                 `json:"status"`
	Data        map[string]interface{} `json:"data"`
	ProcessedAt time.Time              `json:"processed_at"`
	Metadata    map[string]string      `json:"metadata"`
	Error       string                 `json:"error,omitempty"`
}

// OrderCommand represents an order command
type OrderCommand struct {
	CommandID   string            `json:"command_id" validate:"required"`
	OrderID     string            `json:"order_id" validate:"required"`
	UserID      string            `json:"user_id" validate:"required"`
	Action      string            `json:"action" validate:"required"`
	Items       []OrderItem       `json:"items" validate:"required"`
	TotalAmount float64           `json:"total_amount" validate:"required,gt=0"`
	Currency    string            `json:"currency" validate:"required,len=3"`
	Metadata    map[string]string `json:"metadata"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// OrderItem represents an order item
type OrderItem struct {
	ProductID string  `json:"product_id" validate:"required"`
	Quantity  int     `json:"quantity" validate:"required,gt=0"`
	Price     float64 `json:"price" validate:"required,gt=0"`
	Name      string  `json:"name" validate:"required"`
}

// OrderCommandResult represents the result of order command execution
type OrderCommandResult struct {
	CommandID   string            `json:"command_id"`
	OrderID     string            `json:"order_id"`
	UserID      string            `json:"user_id"`
	Action      string            `json:"action"`
	Status      string            `json:"status"`
	Items       []OrderItem       `json:"items"`
	TotalAmount float64           `json:"total_amount"`
	Currency    string            `json:"currency"`
	ProcessedAt time.Time         `json:"processed_at"`
	Metadata    map[string]string `json:"metadata"`
	Error       string            `json:"error,omitempty"`
}

// NotificationCommand represents a notification command
type NotificationCommand struct {
	CommandID string            `json:"command_id" validate:"required"`
	UserID    string            `json:"user_id" validate:"required"`
	Channel   string            `json:"channel" validate:"required"`
	Type      string            `json:"type" validate:"required"`
	Title     string            `json:"title" validate:"required"`
	Message   string            `json:"message" validate:"required"`
	Priority  string            `json:"priority"`
	Metadata  map[string]string `json:"metadata"`
	CreatedAt time.Time         `json:"created_at"`
	UpdatedAt time.Time         `json:"updated_at"`
}

// NotificationCommandResult represents the result of notification command execution
type NotificationCommandResult struct {
	CommandID      string            `json:"command_id"`
	NotificationID string            `json:"notification_id"`
	UserID         string            `json:"user_id"`
	Channel        string            `json:"channel"`
	Type           string            `json:"type"`
	Status         string            `json:"status"`
	SentAt         time.Time         `json:"sent_at"`
	DeliveryID     string            `json:"delivery_id"`
	Metadata       map[string]string `json:"metadata"`
	Error          string            `json:"error,omitempty"`
}

// InventoryCommand represents an inventory command
type InventoryCommand struct {
	CommandID string            `json:"command_id" validate:"required"`
	ProductID string            `json:"product_id" validate:"required"`
	Action    string            `json:"action" validate:"required"`
	Quantity  int               `json:"quantity" validate:"required"`
	Reason    string            `json:"reason"`
	Metadata  map[string]string `json:"metadata"`
	CreatedAt time.Time         `json:"created_at"`
	UpdatedAt time.Time         `json:"updated_at"`
}

// InventoryCommandResult represents the result of inventory command execution
type InventoryCommandResult struct {
	CommandID   string            `json:"command_id"`
	ProductID   string            `json:"product_id"`
	Action      string            `json:"action"`
	Status      string            `json:"status"`
	Quantity    int               `json:"quantity"`
	NewStock    int               `json:"new_stock"`
	Reason      string            `json:"reason"`
	ProcessedAt time.Time         `json:"processed_at"`
	Metadata    map[string]string `json:"metadata"`
	Error       string            `json:"error,omitempty"`
}

// RefundCommand represents a refund command
type RefundCommand struct {
	CommandID string            `json:"command_id" validate:"required"`
	PaymentID string            `json:"payment_id" validate:"required"`
	UserID    string            `json:"user_id" validate:"required"`
	Amount    float64           `json:"amount" validate:"required,gt=0"`
	Currency  string            `json:"currency" validate:"required,len=3"`
	Reason    string            `json:"reason" validate:"required"`
	Metadata  map[string]string `json:"metadata"`
	CreatedAt time.Time         `json:"created_at"`
	UpdatedAt time.Time         `json:"updated_at"`
}

// RefundCommandResult represents the result of refund command execution
type RefundCommandResult struct {
	CommandID   string            `json:"command_id"`
	RefundID    string            `json:"refund_id"`
	PaymentID   string            `json:"payment_id"`
	UserID      string            `json:"user_id"`
	Amount      float64           `json:"amount"`
	Currency    string            `json:"currency"`
	Status      string            `json:"status"`
	Reason      string            `json:"reason"`
	ProcessedAt time.Time         `json:"processed_at"`
	Metadata    map[string]string `json:"metadata"`
	Error       string            `json:"error,omitempty"`
}

// AuditCommand represents an audit command
type AuditCommand struct {
	CommandID  string                 `json:"command_id" validate:"required"`
	EntityType string                 `json:"entity_type" validate:"required"`
	EntityID   string                 `json:"entity_id" validate:"required"`
	Action     string                 `json:"action" validate:"required"`
	Changes    map[string]interface{} `json:"changes" validate:"required"`
	UserID     string                 `json:"user_id"`
	IPAddress  string                 `json:"ip_address"`
	UserAgent  string                 `json:"user_agent"`
	Metadata   map[string]string      `json:"metadata"`
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at"`
}

// AuditCommandResult represents the result of audit command execution
type AuditCommandResult struct {
	CommandID   string                 `json:"command_id"`
	AuditID     string                 `json:"audit_id"`
	EntityType  string                 `json:"entity_type"`
	EntityID    string                 `json:"entity_id"`
	Action      string                 `json:"action"`
	Status      string                 `json:"status"`
	Changes     map[string]interface{} `json:"changes"`
	UserID      string                 `json:"user_id"`
	IPAddress   string                 `json:"ip_address"`
	UserAgent   string                 `json:"user_agent"`
	ProcessedAt time.Time              `json:"processed_at"`
	Metadata    map[string]string      `json:"metadata"`
	Error       string                 `json:"error,omitempty"`
}

// SystemCommand represents a system command
type SystemCommand struct {
	CommandID  string                 `json:"command_id" validate:"required"`
	Action     string                 `json:"action" validate:"required"`
	Parameters map[string]interface{} `json:"parameters" validate:"required"`
	Metadata   map[string]string      `json:"metadata"`
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at"`
}

// SystemCommandResult represents the result of system command execution
type SystemCommandResult struct {
	CommandID   string            `json:"command_id"`
	Action      string            `json:"action"`
	Status      string            `json:"status"`
	Result      interface{}       `json:"result"`
	ProcessedAt time.Time         `json:"processed_at"`
	Metadata    map[string]string `json:"metadata"`
	Error       string            `json:"error,omitempty"`
}

// CommandRequest represents a generic command request
type CommandRequest struct {
	CommandID   string            `json:"command_id" validate:"required"`
	CommandType string            `json:"command_type" validate:"required"`
	Data        interface{}       `json:"data" validate:"required"`
	Priority    CommandPriority   `json:"priority"`
	Metadata    map[string]string `json:"metadata"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// CommandResponse represents a generic command response
type CommandResponse struct {
	CommandID   string            `json:"command_id"`
	CommandType string            `json:"command_type"`
	Status      string            `json:"status"`
	Result      interface{}       `json:"result"`
	ProcessedAt time.Time         `json:"processed_at"`
	Metadata    map[string]string `json:"metadata"`
	Error       string            `json:"error,omitempty"`
}

// CommandInfo represents information about a command
type CommandInfo struct {
	CommandID   string            `json:"command_id"`
	CommandType string            `json:"command_type"`
	Description string            `json:"description"`
	Status      string            `json:"status"`
	Priority    CommandPriority   `json:"priority"`
	CreatedAt   time.Time         `json:"created_at"`
	ExecutedAt  time.Time         `json:"executed_at"`
	CompletedAt time.Time         `json:"completed_at"`
	Duration    time.Duration     `json:"duration"`
	RetryCount  int               `json:"retry_count"`
	CanUndo     bool              `json:"can_undo"`
	Metadata    map[string]string `json:"metadata"`
}

// CommandHealth represents the health status of a command
type CommandHealth struct {
	CommandType string                 `json:"command_type"`
	Status      string                 `json:"status"`
	Message     string                 `json:"message"`
	LastCheck   time.Time              `json:"last_check"`
	Metrics     map[string]interface{} `json:"metrics"`
	Error       string                 `json:"error,omitempty"`
}

// CommandTrend represents a trend data point for commands
type CommandTrend struct {
	Timestamp   time.Time     `json:"timestamp"`
	CommandType string        `json:"command_type"`
	Count       int           `json:"count"`
	SuccessRate float64       `json:"success_rate"`
	AvgDuration time.Duration `json:"avg_duration"`
}
