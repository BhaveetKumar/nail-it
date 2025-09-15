package models

import "time"

// User represents a user in the system
type User struct {
	ID        string    `json:"id" bson:"id"`
	Email     string    `json:"email" bson:"email"`
	Name      string    `json:"name" bson:"name"`
	CreatedAt time.Time `json:"created_at" bson:"created_at"`
	UpdatedAt time.Time `json:"updated_at" bson:"updated_at"`
}

// Payment represents a payment in the system
type Payment struct {
	ID        string    `json:"id" bson:"id"`
	UserID    string    `json:"user_id" bson:"user_id"`
	Amount    float64   `json:"amount" bson:"amount"`
	Currency  string    `json:"currency" bson:"currency"`
	Status    string    `json:"status" bson:"status"`
	CreatedAt time.Time `json:"created_at" bson:"created_at"`
	UpdatedAt time.Time `json:"updated_at" bson:"updated_at"`
}

// AuditLog represents an audit log entry
type AuditLog struct {
	ID         string                 `json:"id" bson:"id"`
	Action     string                 `json:"action" bson:"action"`
	EntityType string                 `json:"entity_type" bson:"entity_type"`
	EntityID   string                 `json:"entity_id" bson:"entity_id"`
	UserID     string                 `json:"user_id" bson:"user_id"`
	Details    map[string]interface{} `json:"details" bson:"details"`
	CreatedAt  time.Time              `json:"created_at" bson:"created_at"`
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
	UserID   string  `json:"user_id" binding:"required"`
	Amount   float64 `json:"amount" binding:"required,gt=0"`
	Currency string  `json:"currency" binding:"required,len=3"`
}

// UpdatePaymentStatusRequest represents the request to update payment status
type UpdatePaymentStatusRequest struct {
	Status string `json:"status" binding:"required,oneof=pending processing completed failed"`
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

// UserEvent represents a user-related event
type UserEvent struct {
	Type      string      `json:"type"`
	UserID    string      `json:"user_id"`
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
