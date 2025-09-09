package event_sourcing

import (
	"time"
)

// UserCreatedEvent represents a user created event
type UserCreatedEvent struct {
	*ConcreteEvent
	UserID    string `json:"user_id"`
	Email     string `json:"email"`
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
	Role      string `json:"role"`
}

// NewUserCreatedEvent creates a new user created event
func NewUserCreatedEvent(aggregateID, userID, email, firstName, lastName, role string) *UserCreatedEvent {
	return &UserCreatedEvent{
		ConcreteEvent: &ConcreteEvent{
			ID:            generateID(),
			Type:          "user_created",
			AggregateID:   aggregateID,
			AggregateType: "user",
			Version:       1,
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		UserID:    userID,
		Email:     email,
		FirstName: firstName,
		LastName:  lastName,
		Role:      role,
	}
}

// GetUserID returns the user ID
func (uce *UserCreatedEvent) GetUserID() string {
	return uce.UserID
}

// GetEmail returns the user email
func (uce *UserCreatedEvent) GetEmail() string {
	return uce.Email
}

// GetFirstName returns the user first name
func (uce *UserCreatedEvent) GetFirstName() string {
	return uce.FirstName
}

// GetLastName returns the user last name
func (uce *UserCreatedEvent) GetLastName() string {
	return uce.LastName
}

// GetRole returns the user role
func (uce *UserCreatedEvent) GetRole() string {
	return uce.Role
}

// UserUpdatedEvent represents a user updated event
type UserUpdatedEvent struct {
	*ConcreteEvent
	UserID    string                 `json:"user_id"`
	Changes   map[string]interface{} `json:"changes"`
	UpdatedBy string                 `json:"updated_by"`
}

// NewUserUpdatedEvent creates a new user updated event
func NewUserUpdatedEvent(aggregateID, userID, updatedBy string, changes map[string]interface{}) *UserUpdatedEvent {
	return &UserUpdatedEvent{
		ConcreteEvent: &ConcreteEvent{
			ID:            generateID(),
			Type:          "user_updated",
			AggregateID:   aggregateID,
			AggregateType: "user",
			Version:       1,
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		UserID:    userID,
		Changes:   changes,
		UpdatedBy: updatedBy,
	}
}

// GetUserID returns the user ID
func (uue *UserUpdatedEvent) GetUserID() string {
	return uue.UserID
}

// GetChanges returns the changes
func (uue *UserUpdatedEvent) GetChanges() map[string]interface{} {
	return uue.Changes
}

// GetUpdatedBy returns who updated the user
func (uue *UserUpdatedEvent) GetUpdatedBy() string {
	return uue.UpdatedBy
}

// UserDeletedEvent represents a user deleted event
type UserDeletedEvent struct {
	*ConcreteEvent
	UserID    string `json:"user_id"`
	DeletedBy string `json:"deleted_by"`
	Reason    string `json:"reason"`
}

// NewUserDeletedEvent creates a new user deleted event
func NewUserDeletedEvent(aggregateID, userID, deletedBy, reason string) *UserDeletedEvent {
	return &UserDeletedEvent{
		ConcreteEvent: &ConcreteEvent{
			ID:            generateID(),
			Type:          "user_deleted",
			AggregateID:   aggregateID,
			AggregateType: "user",
			Version:       1,
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		UserID:    userID,
		DeletedBy: deletedBy,
		Reason:    reason,
	}
}

// GetUserID returns the user ID
func (ude *UserDeletedEvent) GetUserID() string {
	return ude.UserID
}

// GetDeletedBy returns who deleted the user
func (ude *UserDeletedEvent) GetDeletedBy() string {
	return ude.DeletedBy
}

// GetReason returns the deletion reason
func (ude *UserDeletedEvent) GetReason() string {
	return ude.Reason
}

// OrderCreatedEvent represents an order created event
type OrderCreatedEvent struct {
	*ConcreteEvent
	OrderID     string  `json:"order_id"`
	UserID      string  `json:"user_id"`
	OrderNumber string  `json:"order_number"`
	TotalAmount float64 `json:"total_amount"`
	Currency    string  `json:"currency"`
	Status      string  `json:"status"`
}

// NewOrderCreatedEvent creates a new order created event
func NewOrderCreatedEvent(aggregateID, orderID, userID, orderNumber string, totalAmount float64, currency string) *OrderCreatedEvent {
	return &OrderCreatedEvent{
		ConcreteEvent: &ConcreteEvent{
			ID:            generateID(),
			Type:          "order_created",
			AggregateID:   aggregateID,
			AggregateType: "order",
			Version:       1,
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		OrderID:     orderID,
		UserID:      userID,
		OrderNumber: orderNumber,
		TotalAmount: totalAmount,
		Currency:    currency,
		Status:      "pending",
	}
}

// GetOrderID returns the order ID
func (oce *OrderCreatedEvent) GetOrderID() string {
	return oce.OrderID
}

// GetUserID returns the user ID
func (oce *OrderCreatedEvent) GetUserID() string {
	return oce.UserID
}

// GetOrderNumber returns the order number
func (oce *OrderCreatedEvent) GetOrderNumber() string {
	return oce.OrderNumber
}

// GetTotalAmount returns the total amount
func (oce *OrderCreatedEvent) GetTotalAmount() float64 {
	return oce.TotalAmount
}

// GetCurrency returns the currency
func (oce *OrderCreatedEvent) GetCurrency() string {
	return oce.Currency
}

// GetStatus returns the order status
func (oce *OrderCreatedEvent) GetStatus() string {
	return oce.Status
}

// OrderStatusChangedEvent represents an order status changed event
type OrderStatusChangedEvent struct {
	*ConcreteEvent
	OrderID      string `json:"order_id"`
	OldStatus    string `json:"old_status"`
	NewStatus    string `json:"new_status"`
	ChangedBy    string `json:"changed_by"`
	Reason       string `json:"reason"`
	ChangedAt    time.Time `json:"changed_at"`
}

// NewOrderStatusChangedEvent creates a new order status changed event
func NewOrderStatusChangedEvent(aggregateID, orderID, oldStatus, newStatus, changedBy, reason string) *OrderStatusChangedEvent {
	return &OrderStatusChangedEvent{
		ConcreteEvent: &ConcreteEvent{
			ID:            generateID(),
			Type:          "order_status_changed",
			AggregateID:   aggregateID,
			AggregateType: "order",
			Version:       1,
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		OrderID:      orderID,
		OldStatus:    oldStatus,
		NewStatus:    newStatus,
		ChangedBy:    changedBy,
		Reason:       reason,
		ChangedAt:    time.Now(),
	}
}

// GetOrderID returns the order ID
func (osce *OrderStatusChangedEvent) GetOrderID() string {
	return osce.OrderID
}

// GetOldStatus returns the old status
func (osce *OrderStatusChangedEvent) GetOldStatus() string {
	return osce.OldStatus
}

// GetNewStatus returns the new status
func (osce *OrderStatusChangedEvent) GetNewStatus() string {
	return osce.NewStatus
}

// GetChangedBy returns who changed the status
func (osce *OrderStatusChangedEvent) GetChangedBy() string {
	return osce.ChangedBy
}

// GetReason returns the change reason
func (osce *OrderStatusChangedEvent) GetReason() string {
	return osce.Reason
}

// GetChangedAt returns when the status was changed
func (osce *OrderStatusChangedEvent) GetChangedAt() time.Time {
	return osce.ChangedAt
}

// PaymentProcessedEvent represents a payment processed event
type PaymentProcessedEvent struct {
	*ConcreteEvent
	PaymentID     string  `json:"payment_id"`
	OrderID       string  `json:"order_id"`
	UserID        string  `json:"user_id"`
	Amount        float64 `json:"amount"`
	Currency      string  `json:"currency"`
	PaymentMethod string  `json:"payment_method"`
	Status        string  `json:"status"`
	TransactionID string  `json:"transaction_id"`
	Gateway       string  `json:"gateway"`
}

// NewPaymentProcessedEvent creates a new payment processed event
func NewPaymentProcessedEvent(aggregateID, paymentID, orderID, userID string, amount float64, currency, paymentMethod, status, transactionID, gateway string) *PaymentProcessedEvent {
	return &PaymentProcessedEvent{
		ConcreteEvent: &ConcreteEvent{
			ID:            generateID(),
			Type:          "payment_processed",
			AggregateID:   aggregateID,
			AggregateType: "payment",
			Version:       1,
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		PaymentID:     paymentID,
		OrderID:       orderID,
		UserID:        userID,
		Amount:        amount,
		Currency:      currency,
		PaymentMethod: paymentMethod,
		Status:        status,
		TransactionID: transactionID,
		Gateway:       gateway,
	}
}

// GetPaymentID returns the payment ID
func (ppe *PaymentProcessedEvent) GetPaymentID() string {
	return ppe.PaymentID
}

// GetOrderID returns the order ID
func (ppe *PaymentProcessedEvent) GetOrderID() string {
	return ppe.OrderID
}

// GetUserID returns the user ID
func (ppe *PaymentProcessedEvent) GetUserID() string {
	return ppe.UserID
}

// GetAmount returns the payment amount
func (ppe *PaymentProcessedEvent) GetAmount() float64 {
	return ppe.Amount
}

// GetCurrency returns the currency
func (ppe *PaymentProcessedEvent) GetCurrency() string {
	return ppe.Currency
}

// GetPaymentMethod returns the payment method
func (ppe *PaymentProcessedEvent) GetPaymentMethod() string {
	return ppe.PaymentMethod
}

// GetStatus returns the payment status
func (ppe *PaymentProcessedEvent) GetStatus() string {
	return ppe.Status
}

// GetTransactionID returns the transaction ID
func (ppe *PaymentProcessedEvent) GetTransactionID() string {
	return ppe.TransactionID
}

// GetGateway returns the payment gateway
func (ppe *PaymentProcessedEvent) GetGateway() string {
	return ppe.Gateway
}

// UserAggregate represents a user aggregate
type UserAggregate struct {
	*ConcreteAggregate
	UserID    string `json:"user_id"`
	Email     string `json:"email"`
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
	Role      string `json:"role"`
	Status    string `json:"status"`
}

// NewUserAggregate creates a new user aggregate
func NewUserAggregate(aggregateID, userID, email, firstName, lastName, role string) *UserAggregate {
	return &UserAggregate{
		ConcreteAggregate: &ConcreteAggregate{
			ID:                aggregateID,
			Type:              "user",
			Version:           0,
			Events:            make([]Event, 0),
			UncommittedEvents: make([]Event, 0),
			State:             make(map[string]interface{}),
			CreatedAt:         time.Now(),
			UpdatedAt:         time.Now(),
			Active:            true,
			Metadata:          make(map[string]interface{}),
		},
		UserID:    userID,
		Email:     email,
		FirstName: firstName,
		LastName:  lastName,
		Role:      role,
		Status:    "active",
	}
}

// GetUserID returns the user ID
func (ua *UserAggregate) GetUserID() string {
	return ua.UserID
}

// GetEmail returns the user email
func (ua *UserAggregate) GetEmail() string {
	return ua.Email
}

// SetEmail sets the user email
func (ua *UserAggregate) SetEmail(email string) {
	ua.Email = email
	ua.UpdatedAt = time.Now()
}

// GetFirstName returns the user first name
func (ua *UserAggregate) GetFirstName() string {
	return ua.FirstName
}

// SetFirstName sets the user first name
func (ua *UserAggregate) SetFirstName(firstName string) {
	ua.FirstName = firstName
	ua.UpdatedAt = time.Now()
}

// GetLastName returns the user last name
func (ua *UserAggregate) GetLastName() string {
	return ua.LastName
}

// SetLastName sets the user last name
func (ua *UserAggregate) SetLastName(lastName string) {
	ua.LastName = lastName
	ua.UpdatedAt = time.Now()
}

// GetRole returns the user role
func (ua *UserAggregate) GetRole() string {
	return ua.Role
}

// SetRole sets the user role
func (ua *UserAggregate) SetRole(role string) {
	ua.Role = role
	ua.UpdatedAt = time.Now()
}

// GetStatus returns the user status
func (ua *UserAggregate) GetStatus() string {
	return ua.Status
}

// SetStatus sets the user status
func (ua *UserAggregate) SetStatus(status string) {
	ua.Status = status
	ua.UpdatedAt = time.Now()
}

// OrderAggregate represents an order aggregate
type OrderAggregate struct {
	*ConcreteAggregate
	OrderID     string  `json:"order_id"`
	UserID      string  `json:"user_id"`
	OrderNumber string  `json:"order_number"`
	TotalAmount float64 `json:"total_amount"`
	Currency    string  `json:"currency"`
	Status      string  `json:"status"`
	OrderDate   time.Time `json:"order_date"`
}

// NewOrderAggregate creates a new order aggregate
func NewOrderAggregate(aggregateID, orderID, userID, orderNumber string, totalAmount float64, currency string) *OrderAggregate {
	return &OrderAggregate{
		ConcreteAggregate: &ConcreteAggregate{
			ID:                aggregateID,
			Type:              "order",
			Version:           0,
			Events:            make([]Event, 0),
			UncommittedEvents: make([]Event, 0),
			State:             make(map[string]interface{}),
			CreatedAt:         time.Now(),
			UpdatedAt:         time.Now(),
			Active:            true,
			Metadata:          make(map[string]interface{}),
		},
		OrderID:     orderID,
		UserID:      userID,
		OrderNumber: orderNumber,
		TotalAmount: totalAmount,
		Currency:    currency,
		Status:      "pending",
		OrderDate:   time.Now(),
	}
}

// GetOrderID returns the order ID
func (oa *OrderAggregate) GetOrderID() string {
	return oa.OrderID
}

// GetUserID returns the user ID
func (oa *OrderAggregate) GetUserID() string {
	return oa.UserID
}

// GetOrderNumber returns the order number
func (oa *OrderAggregate) GetOrderNumber() string {
	return oa.OrderNumber
}

// GetTotalAmount returns the total amount
func (oa *OrderAggregate) GetTotalAmount() float64 {
	return oa.TotalAmount
}

// SetTotalAmount sets the total amount
func (oa *OrderAggregate) SetTotalAmount(totalAmount float64) {
	oa.TotalAmount = totalAmount
	oa.UpdatedAt = time.Now()
}

// GetCurrency returns the currency
func (oa *OrderAggregate) GetCurrency() string {
	return oa.Currency
}

// GetStatus returns the order status
func (oa *OrderAggregate) GetStatus() string {
	return oa.Status
}

// SetStatus sets the order status
func (oa *OrderAggregate) SetStatus(status string) {
	oa.Status = status
	oa.UpdatedAt = time.Now()
}

// GetOrderDate returns the order date
func (oa *OrderAggregate) GetOrderDate() time.Time {
	return oa.OrderDate
}

// PaymentAggregate represents a payment aggregate
type PaymentAggregate struct {
	*ConcreteAggregate
	PaymentID     string  `json:"payment_id"`
	OrderID       string  `json:"order_id"`
	UserID        string  `json:"user_id"`
	Amount        float64 `json:"amount"`
	Currency      string  `json:"currency"`
	PaymentMethod string  `json:"payment_method"`
	Status        string  `json:"status"`
	TransactionID string  `json:"transaction_id"`
	Gateway       string  `json:"gateway"`
	PaymentDate   time.Time `json:"payment_date"`
}

// NewPaymentAggregate creates a new payment aggregate
func NewPaymentAggregate(aggregateID, paymentID, orderID, userID string, amount float64, currency, paymentMethod, gateway string) *PaymentAggregate {
	return &PaymentAggregate{
		ConcreteAggregate: &ConcreteAggregate{
			ID:                aggregateID,
			Type:              "payment",
			Version:           0,
			Events:            make([]Event, 0),
			UncommittedEvents: make([]Event, 0),
			State:             make(map[string]interface{}),
			CreatedAt:         time.Now(),
			UpdatedAt:         time.Now(),
			Active:            true,
			Metadata:          make(map[string]interface{}),
		},
		PaymentID:     paymentID,
		OrderID:       orderID,
		UserID:        userID,
		Amount:        amount,
		Currency:      currency,
		PaymentMethod: paymentMethod,
		Status:        "pending",
		TransactionID: "",
		Gateway:       gateway,
		PaymentDate:   time.Time{},
	}
}

// GetPaymentID returns the payment ID
func (pa *PaymentAggregate) GetPaymentID() string {
	return pa.PaymentID
}

// GetOrderID returns the order ID
func (pa *PaymentAggregate) GetOrderID() string {
	return pa.OrderID
}

// GetUserID returns the user ID
func (pa *PaymentAggregate) GetUserID() string {
	return pa.UserID
}

// GetAmount returns the payment amount
func (pa *PaymentAggregate) GetAmount() float64 {
	return pa.Amount
}

// GetCurrency returns the currency
func (pa *PaymentAggregate) GetCurrency() string {
	return pa.Currency
}

// GetPaymentMethod returns the payment method
func (pa *PaymentAggregate) GetPaymentMethod() string {
	return pa.PaymentMethod
}

// GetStatus returns the payment status
func (pa *PaymentAggregate) GetStatus() string {
	return pa.Status
}

// SetStatus sets the payment status
func (pa *PaymentAggregate) SetStatus(status string) {
	pa.Status = status
	pa.UpdatedAt = time.Now()
}

// GetTransactionID returns the transaction ID
func (pa *PaymentAggregate) GetTransactionID() string {
	return pa.TransactionID
}

// SetTransactionID sets the transaction ID
func (pa *PaymentAggregate) SetTransactionID(transactionID string) {
	pa.TransactionID = transactionID
	pa.UpdatedAt = time.Now()
}

// GetGateway returns the payment gateway
func (pa *PaymentAggregate) GetGateway() string {
	return pa.Gateway
}

// GetPaymentDate returns the payment date
func (pa *PaymentAggregate) GetPaymentDate() time.Time {
	return pa.PaymentDate
}

// SetPaymentDate sets the payment date
func (pa *PaymentAggregate) SetPaymentDate(paymentDate time.Time) {
	pa.PaymentDate = paymentDate
	pa.UpdatedAt = time.Now()
}
