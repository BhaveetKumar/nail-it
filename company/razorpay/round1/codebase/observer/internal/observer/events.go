package observer

import (
	"time"

	"github.com/google/uuid"
)

// BaseEvent represents a base event implementation
type BaseEvent struct {
	ID        string      `json:"id"`
	Type      string      `json:"type"`
	Timestamp time.Time   `json:"timestamp"`
	Data      interface{} `json:"data"`
	Source    string      `json:"source"`
	Version   string      `json:"version"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// NewBaseEvent creates a new base event
func NewBaseEvent(eventType string, data interface{}, source string) *BaseEvent {
	return &BaseEvent{
		ID:        uuid.New().String(),
		Type:      eventType,
		Timestamp: time.Now(),
		Data:      data,
		Source:    source,
		Version:   "1.0",
		Metadata:  make(map[string]interface{}),
	}
}

func (e *BaseEvent) GetType() string {
	return e.Type
}

func (e *BaseEvent) GetID() string {
	return e.ID
}

func (e *BaseEvent) GetTimestamp() time.Time {
	return e.Timestamp
}

func (e *BaseEvent) GetData() interface{} {
	return e.Data
}

func (e *BaseEvent) GetSource() string {
	return e.Source
}

func (e *BaseEvent) GetVersion() string {
	return e.Version
}

func (e *BaseEvent) GetMetadata() map[string]interface{} {
	return e.Metadata
}

func (e *BaseEvent) SetMetadata(key string, value interface{}) {
	if e.Metadata == nil {
		e.Metadata = make(map[string]interface{})
	}
	e.Metadata[key] = value
}

func (e *BaseEvent) GetMetadataValue(key string) (interface{}, bool) {
	if e.Metadata == nil {
		return nil, false
	}
	value, exists := e.Metadata[key]
	return value, exists
}

// PaymentEvent represents a payment-related event
type PaymentEvent struct {
	*BaseEvent
	PaymentID    string  `json:"payment_id"`
	UserID       string  `json:"user_id"`
	Amount       float64 `json:"amount"`
	Currency     string  `json:"currency"`
	Status       string  `json:"status"`
	Gateway      string  `json:"gateway"`
	TransactionID string `json:"transaction_id,omitempty"`
}

// NewPaymentEvent creates a new payment event
func NewPaymentEvent(eventType string, paymentID, userID string, amount float64, currency, status, gateway string) *PaymentEvent {
	baseEvent := NewBaseEvent(eventType, nil, "payment-service")
	
	paymentData := map[string]interface{}{
		"payment_id":     paymentID,
		"user_id":        userID,
		"amount":         amount,
		"currency":       currency,
		"status":         status,
		"gateway":        gateway,
	}
	
	baseEvent.Data = paymentData
	
	return &PaymentEvent{
		BaseEvent:    baseEvent,
		PaymentID:    paymentID,
		UserID:       userID,
		Amount:       amount,
		Currency:     currency,
		Status:       status,
		Gateway:      gateway,
	}
}

// UserEvent represents a user-related event
type UserEvent struct {
	*BaseEvent
	UserID    string `json:"user_id"`
	Email     string `json:"email"`
	Name      string `json:"name"`
	Status    string `json:"status"`
	Action    string `json:"action"`
}

// NewUserEvent creates a new user event
func NewUserEvent(eventType string, userID, email, name, status, action string) *UserEvent {
	baseEvent := NewBaseEvent(eventType, nil, "user-service")
	
	userData := map[string]interface{}{
		"user_id": userID,
		"email":   email,
		"name":    name,
		"status":  status,
		"action":  action,
	}
	
	baseEvent.Data = userData
	
	return &UserEvent{
		BaseEvent: baseEvent,
		UserID:    userID,
		Email:     email,
		Name:      name,
		Status:    status,
		Action:    action,
	}
}

// OrderEvent represents an order-related event
type OrderEvent struct {
	*BaseEvent
	OrderID     string  `json:"order_id"`
	UserID      string  `json:"user_id"`
	PaymentID   string  `json:"payment_id,omitempty"`
	TotalAmount float64 `json:"total_amount"`
	Currency    string  `json:"currency"`
	Status      string  `json:"status"`
	Items       []OrderItem `json:"items"`
}

// OrderItem represents an order item
type OrderItem struct {
	ProductID string  `json:"product_id"`
	Quantity  int     `json:"quantity"`
	Price     float64 `json:"price"`
	Total     float64 `json:"total"`
}

// NewOrderEvent creates a new order event
func NewOrderEvent(eventType string, orderID, userID string, totalAmount float64, currency, status string, items []OrderItem) *OrderEvent {
	baseEvent := NewBaseEvent(eventType, nil, "order-service")
	
	orderData := map[string]interface{}{
		"order_id":      orderID,
		"user_id":       userID,
		"total_amount":  totalAmount,
		"currency":      currency,
		"status":        status,
		"items":         items,
	}
	
	baseEvent.Data = orderData
	
	return &OrderEvent{
		BaseEvent:   baseEvent,
		OrderID:     orderID,
		UserID:      userID,
		TotalAmount: totalAmount,
		Currency:    currency,
		Status:      status,
		Items:       items,
	}
}

// ProductEvent represents a product-related event
type ProductEvent struct {
	*BaseEvent
	ProductID   string  `json:"product_id"`
	Name        string  `json:"name"`
	Category    string  `json:"category"`
	Price       float64 `json:"price"`
	Currency    string  `json:"currency"`
	Stock       int     `json:"stock"`
	Status      string  `json:"status"`
	Action      string  `json:"action"`
}

// NewProductEvent creates a new product event
func NewProductEvent(eventType string, productID, name, category string, price float64, currency string, stock int, status, action string) *ProductEvent {
	baseEvent := NewBaseEvent(eventType, nil, "product-service")
	
	productData := map[string]interface{}{
		"product_id": productID,
		"name":       name,
		"category":   category,
		"price":      price,
		"currency":   currency,
		"stock":      stock,
		"status":     status,
		"action":     action,
	}
	
	baseEvent.Data = productData
	
	return &ProductEvent{
		BaseEvent: baseEvent,
		ProductID: productID,
		Name:      name,
		Category:  category,
		Price:     price,
		Currency:  currency,
		Stock:     stock,
		Status:    status,
		Action:    action,
	}
}

// NotificationEvent represents a notification-related event
type NotificationEvent struct {
	*BaseEvent
	NotificationID string `json:"notification_id"`
	UserID         string `json:"user_id"`
	Channel        string `json:"channel"`
	Type           string `json:"type"`
	Subject        string `json:"subject,omitempty"`
	Message        string `json:"message"`
	Status         string `json:"status"`
	SentAt         *time.Time `json:"sent_at,omitempty"`
}

// NewNotificationEvent creates a new notification event
func NewNotificationEvent(eventType string, notificationID, userID, channel, notificationType, subject, message, status string) *NotificationEvent {
	baseEvent := NewBaseEvent(eventType, nil, "notification-service")
	
	notificationData := map[string]interface{}{
		"notification_id": notificationID,
		"user_id":         userID,
		"channel":         channel,
		"type":            notificationType,
		"subject":         subject,
		"message":         message,
		"status":          status,
	}
	
	baseEvent.Data = notificationData
	
	return &NotificationEvent{
		BaseEvent:      baseEvent,
		NotificationID: notificationID,
		UserID:         userID,
		Channel:        channel,
		Type:           notificationType,
		Subject:        subject,
		Message:        message,
		Status:         status,
	}
}

// AuditEvent represents an audit-related event
type AuditEvent struct {
	*BaseEvent
	EntityType string                 `json:"entity_type"`
	EntityID   string                 `json:"entity_id"`
	Action     string                 `json:"action"`
	UserID     string                 `json:"user_id,omitempty"`
	Changes    map[string]interface{} `json:"changes,omitempty"`
	IPAddress  string                 `json:"ip_address,omitempty"`
	UserAgent  string                 `json:"user_agent,omitempty"`
}

// NewAuditEvent creates a new audit event
func NewAuditEvent(eventType string, entityType, entityID, action, userID string, changes map[string]interface{}) *AuditEvent {
	baseEvent := NewBaseEvent(eventType, nil, "audit-service")
	
	auditData := map[string]interface{}{
		"entity_type": entityType,
		"entity_id":   entityID,
		"action":      action,
		"user_id":     userID,
		"changes":     changes,
	}
	
	baseEvent.Data = auditData
	
	return &AuditEvent{
		BaseEvent:  baseEvent,
		EntityType: entityType,
		EntityID:   entityID,
		Action:     action,
		UserID:     userID,
		Changes:    changes,
	}
}

// SystemEvent represents a system-related event
type SystemEvent struct {
	*BaseEvent
	Component string                 `json:"component"`
	Level     string                 `json:"level"` // "info", "warn", "error", "fatal"
	Message   string                 `json:"message"`
	Details   map[string]interface{} `json:"details,omitempty"`
}

// NewSystemEvent creates a new system event
func NewSystemEvent(eventType string, component, level, message string, details map[string]interface{}) *SystemEvent {
	baseEvent := NewBaseEvent(eventType, nil, "system-service")
	
	systemData := map[string]interface{}{
		"component": component,
		"level":     level,
		"message":   message,
		"details":   details,
	}
	
	baseEvent.Data = systemData
	
	return &SystemEvent{
		BaseEvent: baseEvent,
		Component: component,
		Level:     level,
		Message:   message,
		Details:   details,
	}
}

// Event constants
const (
	// Payment events
	EventTypePaymentCreated   = "payment.created"
	EventTypePaymentUpdated   = "payment.updated"
	EventTypePaymentCompleted = "payment.completed"
	EventTypePaymentFailed    = "payment.failed"
	EventTypePaymentRefunded  = "payment.refunded"
	
	// User events
	EventTypeUserCreated      = "user.created"
	EventTypeUserUpdated      = "user.updated"
	EventTypeUserDeleted      = "user.deleted"
	EventTypeUserActivated    = "user.activated"
	EventTypeUserDeactivated  = "user.deactivated"
	
	// Order events
	EventTypeOrderCreated     = "order.created"
	EventTypeOrderUpdated     = "order.updated"
	EventTypeOrderCancelled   = "order.cancelled"
	EventTypeOrderCompleted   = "order.completed"
	EventTypeOrderShipped     = "order.shipped"
	EventTypeOrderDelivered   = "order.delivered"
	
	// Product events
	EventTypeProductCreated   = "product.created"
	EventTypeProductUpdated   = "product.updated"
	EventTypeProductDeleted   = "product.deleted"
	EventTypeProductStockLow  = "product.stock_low"
	EventTypeProductOutOfStock = "product.out_of_stock"
	
	// Notification events
	EventTypeNotificationSent     = "notification.sent"
	EventTypeNotificationFailed   = "notification.failed"
	EventTypeNotificationDelivered = "notification.delivered"
	EventTypeNotificationRead     = "notification.read"
	
	// Audit events
	EventTypeAuditLogCreated = "audit.log_created"
	EventTypeAuditLogUpdated = "audit.log_updated"
	
	// System events
	EventTypeSystemStartup   = "system.startup"
	EventTypeSystemShutdown  = "system.shutdown"
	EventTypeSystemError     = "system.error"
	EventTypeSystemWarning   = "system.warning"
	EventTypeSystemInfo      = "system.info"
)

// Event priorities
const (
	PriorityLow    = 1
	PriorityMedium = 2
	PriorityHigh   = 3
	PriorityCritical = 4
)

// Event categories
const (
	CategoryBusiness = "business"
	CategorySystem   = "system"
	CategorySecurity = "security"
	CategoryAudit    = "audit"
	CategoryMetrics  = "metrics"
)
