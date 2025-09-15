package cqrs

import (
	"time"
)

// CreateUserCommand represents a create user command
type CreateUserCommand struct {
	*ConcreteCommand
	Email     string `json:"email"`
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
	Role      string `json:"role"`
}

// NewCreateUserCommand creates a new create user command
func NewCreateUserCommand(aggregateID, email, firstName, lastName, role string) *CreateUserCommand {
	return &CreateUserCommand{
		ConcreteCommand: &ConcreteCommand{
			ID:            generateID(),
			Type:          "create_user",
			AggregateID:   aggregateID,
			AggregateType: "user",
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		Email:     email,
		FirstName: firstName,
		LastName:  lastName,
		Role:      role,
	}
}

// GetEmail returns the user email
func (cuc *CreateUserCommand) GetEmail() string {
	return cuc.Email
}

// GetFirstName returns the user first name
func (cuc *CreateUserCommand) GetFirstName() string {
	return cuc.FirstName
}

// GetLastName returns the user last name
func (cuc *CreateUserCommand) GetLastName() string {
	return cuc.LastName
}

// GetRole returns the user role
func (cuc *CreateUserCommand) GetRole() string {
	return cuc.Role
}

// UpdateUserCommand represents an update user command
type UpdateUserCommand struct {
	*ConcreteCommand
	UserID    string                 `json:"user_id"`
	Changes   map[string]interface{} `json:"changes"`
	UpdatedBy string                 `json:"updated_by"`
}

// NewUpdateUserCommand creates a new update user command
func NewUpdateUserCommand(aggregateID, userID, updatedBy string, changes map[string]interface{}) *UpdateUserCommand {
	return &UpdateUserCommand{
		ConcreteCommand: &ConcreteCommand{
			ID:            generateID(),
			Type:          "update_user",
			AggregateID:   aggregateID,
			AggregateType: "user",
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
func (uuc *UpdateUserCommand) GetUserID() string {
	return uuc.UserID
}

// GetChanges returns the changes
func (uuc *UpdateUserCommand) GetChanges() map[string]interface{} {
	return uuc.Changes
}

// GetUpdatedBy returns who updated the user
func (uuc *UpdateUserCommand) GetUpdatedBy() string {
	return uuc.UpdatedBy
}

// DeleteUserCommand represents a delete user command
type DeleteUserCommand struct {
	*ConcreteCommand
	UserID    string `json:"user_id"`
	DeletedBy string `json:"deleted_by"`
	Reason    string `json:"reason"`
}

// NewDeleteUserCommand creates a new delete user command
func NewDeleteUserCommand(aggregateID, userID, deletedBy, reason string) *DeleteUserCommand {
	return &DeleteUserCommand{
		ConcreteCommand: &ConcreteCommand{
			ID:            generateID(),
			Type:          "delete_user",
			AggregateID:   aggregateID,
			AggregateType: "user",
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
func (duc *DeleteUserCommand) GetUserID() string {
	return duc.UserID
}

// GetDeletedBy returns who deleted the user
func (duc *DeleteUserCommand) GetDeletedBy() string {
	return duc.DeletedBy
}

// GetReason returns the deletion reason
func (duc *DeleteUserCommand) GetReason() string {
	return duc.Reason
}

// CreateOrderCommand represents a create order command
type CreateOrderCommand struct {
	*ConcreteCommand
	OrderID     string      `json:"order_id"`
	UserID      string      `json:"user_id"`
	OrderNumber string      `json:"order_number"`
	TotalAmount float64     `json:"total_amount"`
	Currency    string      `json:"currency"`
	Items       []OrderItem `json:"items"`
}

// OrderItem represents an order item
type OrderItem struct {
	ID          string  `json:"id"`
	ProductID   string  `json:"product_id"`
	ProductName string  `json:"product_name"`
	Quantity    int     `json:"quantity"`
	UnitPrice   float64 `json:"unit_price"`
	TotalPrice  float64 `json:"total_price"`
}

// NewCreateOrderCommand creates a new create order command
func NewCreateOrderCommand(aggregateID, orderID, userID, orderNumber string, totalAmount float64, currency string, items []OrderItem) *CreateOrderCommand {
	return &CreateOrderCommand{
		ConcreteCommand: &ConcreteCommand{
			ID:            generateID(),
			Type:          "create_order",
			AggregateID:   aggregateID,
			AggregateType: "order",
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
		Items:       items,
	}
}

// GetOrderID returns the order ID
func (coc *CreateOrderCommand) GetOrderID() string {
	return coc.OrderID
}

// GetUserID returns the user ID
func (coc *CreateOrderCommand) GetUserID() string {
	return coc.UserID
}

// GetOrderNumber returns the order number
func (coc *CreateOrderCommand) GetOrderNumber() string {
	return coc.OrderNumber
}

// GetTotalAmount returns the total amount
func (coc *CreateOrderCommand) GetTotalAmount() float64 {
	return coc.TotalAmount
}

// GetCurrency returns the currency
func (coc *CreateOrderCommand) GetCurrency() string {
	return coc.Currency
}

// GetItems returns the order items
func (coc *CreateOrderCommand) GetItems() []OrderItem {
	return coc.Items
}

// UpdateOrderStatusCommand represents an update order status command
type UpdateOrderStatusCommand struct {
	*ConcreteCommand
	OrderID   string `json:"order_id"`
	OldStatus string `json:"old_status"`
	NewStatus string `json:"new_status"`
	UpdatedBy string `json:"updated_by"`
	Reason    string `json:"reason"`
}

// NewUpdateOrderStatusCommand creates a new update order status command
func NewUpdateOrderStatusCommand(aggregateID, orderID, oldStatus, newStatus, updatedBy, reason string) *UpdateOrderStatusCommand {
	return &UpdateOrderStatusCommand{
		ConcreteCommand: &ConcreteCommand{
			ID:            generateID(),
			Type:          "update_order_status",
			AggregateID:   aggregateID,
			AggregateType: "order",
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		OrderID:   orderID,
		OldStatus: oldStatus,
		NewStatus: newStatus,
		UpdatedBy: updatedBy,
		Reason:    reason,
	}
}

// GetOrderID returns the order ID
func (uosc *UpdateOrderStatusCommand) GetOrderID() string {
	return uosc.OrderID
}

// GetOldStatus returns the old status
func (uosc *UpdateOrderStatusCommand) GetOldStatus() string {
	return uosc.OldStatus
}

// GetNewStatus returns the new status
func (uosc *UpdateOrderStatusCommand) GetNewStatus() string {
	return uosc.NewStatus
}

// GetUpdatedBy returns who updated the order
func (uosc *UpdateOrderStatusCommand) GetUpdatedBy() string {
	return uosc.UpdatedBy
}

// GetReason returns the update reason
func (uosc *UpdateOrderStatusCommand) GetReason() string {
	return uosc.Reason
}

// ProcessPaymentCommand represents a process payment command
type ProcessPaymentCommand struct {
	*ConcreteCommand
	PaymentID     string  `json:"payment_id"`
	OrderID       string  `json:"order_id"`
	UserID        string  `json:"user_id"`
	Amount        float64 `json:"amount"`
	Currency      string  `json:"currency"`
	PaymentMethod string  `json:"payment_method"`
	Gateway       string  `json:"gateway"`
}

// NewProcessPaymentCommand creates a new process payment command
func NewProcessPaymentCommand(aggregateID, paymentID, orderID, userID string, amount float64, currency, paymentMethod, gateway string) *ProcessPaymentCommand {
	return &ProcessPaymentCommand{
		ConcreteCommand: &ConcreteCommand{
			ID:            generateID(),
			Type:          "process_payment",
			AggregateID:   aggregateID,
			AggregateType: "payment",
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
		Gateway:       gateway,
	}
}

// GetPaymentID returns the payment ID
func (ppc *ProcessPaymentCommand) GetPaymentID() string {
	return ppc.PaymentID
}

// GetOrderID returns the order ID
func (ppc *ProcessPaymentCommand) GetOrderID() string {
	return ppc.OrderID
}

// GetUserID returns the user ID
func (ppc *ProcessPaymentCommand) GetUserID() string {
	return ppc.UserID
}

// GetAmount returns the payment amount
func (ppc *ProcessPaymentCommand) GetAmount() float64 {
	return ppc.Amount
}

// GetCurrency returns the currency
func (ppc *ProcessPaymentCommand) GetCurrency() string {
	return ppc.Currency
}

// GetPaymentMethod returns the payment method
func (ppc *ProcessPaymentCommand) GetPaymentMethod() string {
	return ppc.PaymentMethod
}

// GetGateway returns the payment gateway
func (ppc *ProcessPaymentCommand) GetGateway() string {
	return ppc.Gateway
}

// GetUserQuery represents a get user query
type GetUserQuery struct {
	*ConcreteQuery
	UserID string `json:"user_id"`
}

// NewGetUserQuery creates a new get user query
func NewGetUserQuery(userID string) *GetUserQuery {
	return &GetUserQuery{
		ConcreteQuery: &ConcreteQuery{
			ID:            generateID(),
			Type:          "get_user",
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		UserID: userID,
	}
}

// GetUserID returns the user ID
func (guq *GetUserQuery) GetUserID() string {
	return guq.UserID
}

// GetUsersQuery represents a get users query
type GetUsersQuery struct {
	*ConcreteQuery
	Filters map[string]interface{} `json:"filters"`
	Limit   int                    `json:"limit"`
	Offset  int                    `json:"offset"`
}

// NewGetUsersQuery creates a new get users query
func NewGetUsersQuery(filters map[string]interface{}, limit, offset int) *GetUsersQuery {
	return &GetUsersQuery{
		ConcreteQuery: &ConcreteQuery{
			ID:            generateID(),
			Type:          "get_users",
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		Filters: filters,
		Limit:   limit,
		Offset:  offset,
	}
}

// GetFilters returns the filters
func (guq *GetUsersQuery) GetFilters() map[string]interface{} {
	return guq.Filters
}

// GetLimit returns the limit
func (guq *GetUsersQuery) GetLimit() int {
	return guq.Limit
}

// GetOffset returns the offset
func (guq *GetUsersQuery) GetOffset() int {
	return guq.Offset
}

// GetOrderQuery represents a get order query
type GetOrderQuery struct {
	*ConcreteQuery
	OrderID string `json:"order_id"`
}

// NewGetOrderQuery creates a new get order query
func NewGetOrderQuery(orderID string) *GetOrderQuery {
	return &GetOrderQuery{
		ConcreteQuery: &ConcreteQuery{
			ID:            generateID(),
			Type:          "get_order",
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		OrderID: orderID,
	}
}

// GetOrderID returns the order ID
func (goq *GetOrderQuery) GetOrderID() string {
	return goq.OrderID
}

// GetOrdersQuery represents a get orders query
type GetOrdersQuery struct {
	*ConcreteQuery
	UserID  string                 `json:"user_id"`
	Filters map[string]interface{} `json:"filters"`
	Limit   int                    `json:"limit"`
	Offset  int                    `json:"offset"`
}

// NewGetOrdersQuery creates a new get orders query
func NewGetOrdersQuery(userID string, filters map[string]interface{}, limit, offset int) *GetOrdersQuery {
	return &GetOrdersQuery{
		ConcreteQuery: &ConcreteQuery{
			ID:            generateID(),
			Type:          "get_orders",
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		UserID:  userID,
		Filters: filters,
		Limit:   limit,
		Offset:  offset,
	}
}

// GetUserID returns the user ID
func (goq *GetOrdersQuery) GetUserID() string {
	return goq.UserID
}

// GetFilters returns the filters
func (goq *GetOrdersQuery) GetFilters() map[string]interface{} {
	return goq.Filters
}

// GetLimit returns the limit
func (goq *GetOrdersQuery) GetLimit() int {
	return goq.Limit
}

// GetOffset returns the offset
func (goq *GetOrdersQuery) GetOffset() int {
	return goq.Offset
}

// GetPaymentQuery represents a get payment query
type GetPaymentQuery struct {
	*ConcreteQuery
	PaymentID string `json:"payment_id"`
}

// NewGetPaymentQuery creates a new get payment query
func NewGetPaymentQuery(paymentID string) *GetPaymentQuery {
	return &GetPaymentQuery{
		ConcreteQuery: &ConcreteQuery{
			ID:            generateID(),
			Type:          "get_payment",
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		PaymentID: paymentID,
	}
}

// GetPaymentID returns the payment ID
func (gpq *GetPaymentQuery) GetPaymentID() string {
	return gpq.PaymentID
}

// GetPaymentsQuery represents a get payments query
type GetPaymentsQuery struct {
	*ConcreteQuery
	UserID  string                 `json:"user_id"`
	Filters map[string]interface{} `json:"filters"`
	Limit   int                    `json:"limit"`
	Offset  int                    `json:"offset"`
}

// NewGetPaymentsQuery creates a new get payments query
func NewGetPaymentsQuery(userID string, filters map[string]interface{}, limit, offset int) *GetPaymentsQuery {
	return &GetPaymentsQuery{
		ConcreteQuery: &ConcreteQuery{
			ID:            generateID(),
			Type:          "get_payments",
			Data:          make(map[string]interface{}),
			Metadata:      make(map[string]interface{}),
			Timestamp:     time.Now(),
			CorrelationID: "",
			CausationID:   "",
			Processed:     false,
			ProcessedAt:   time.Time{},
		},
		UserID:  userID,
		Filters: filters,
		Limit:   limit,
		Offset:  offset,
	}
}

// GetUserID returns the user ID
func (gpq *GetPaymentsQuery) GetUserID() string {
	return gpq.UserID
}

// GetFilters returns the filters
func (gpq *GetPaymentsQuery) GetFilters() map[string]interface{} {
	return gpq.Filters
}

// GetLimit returns the limit
func (gpq *GetPaymentsQuery) GetLimit() int {
	return gpq.Limit
}

// GetOffset returns the offset
func (gpq *GetPaymentsQuery) GetOffset() int {
	return gpq.Offset
}

// UserReadModel represents a user read model
type UserReadModel struct {
	*ConcreteReadModel
	UserID    string `json:"user_id"`
	Email     string `json:"email"`
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
	Role      string `json:"role"`
	Status    string `json:"status"`
}

// NewUserReadModel creates a new user read model
func NewUserReadModel(userID, email, firstName, lastName, role, status string) *UserReadModel {
	return &UserReadModel{
		ConcreteReadModel: &ConcreteReadModel{
			ID:        generateID(),
			Type:      "user",
			Data:      make(map[string]interface{}),
			Metadata:  make(map[string]interface{}),
			Version:   1,
			Timestamp: time.Now(),
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
			Active:    true,
		},
		UserID:    userID,
		Email:     email,
		FirstName: firstName,
		LastName:  lastName,
		Role:      role,
		Status:    status,
	}
}

// GetUserID returns the user ID
func (urm *UserReadModel) GetUserID() string {
	return urm.UserID
}

// GetEmail returns the user email
func (urm *UserReadModel) GetEmail() string {
	return urm.Email
}

// GetFirstName returns the user first name
func (urm *UserReadModel) GetFirstName() string {
	return urm.FirstName
}

// GetLastName returns the user last name
func (urm *UserReadModel) GetLastName() string {
	return urm.LastName
}

// GetRole returns the user role
func (urm *UserReadModel) GetRole() string {
	return urm.Role
}

// GetStatus returns the user status
func (urm *UserReadModel) GetStatus() string {
	return urm.Status
}

// OrderReadModel represents an order read model
type OrderReadModel struct {
	*ConcreteReadModel
	OrderID     string      `json:"order_id"`
	UserID      string      `json:"user_id"`
	OrderNumber string      `json:"order_number"`
	TotalAmount float64     `json:"total_amount"`
	Currency    string      `json:"currency"`
	Status      string      `json:"status"`
	OrderDate   time.Time   `json:"order_date"`
	Items       []OrderItem `json:"items"`
}

// NewOrderReadModel creates a new order read model
func NewOrderReadModel(orderID, userID, orderNumber string, totalAmount float64, currency, status string, orderDate time.Time, items []OrderItem) *OrderReadModel {
	return &OrderReadModel{
		ConcreteReadModel: &ConcreteReadModel{
			ID:        generateID(),
			Type:      "order",
			Data:      make(map[string]interface{}),
			Metadata:  make(map[string]interface{}),
			Version:   1,
			Timestamp: time.Now(),
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
			Active:    true,
		},
		OrderID:     orderID,
		UserID:      userID,
		OrderNumber: orderNumber,
		TotalAmount: totalAmount,
		Currency:    currency,
		Status:      status,
		OrderDate:   orderDate,
		Items:       items,
	}
}

// GetOrderID returns the order ID
func (orm *OrderReadModel) GetOrderID() string {
	return orm.OrderID
}

// GetUserID returns the user ID
func (orm *OrderReadModel) GetUserID() string {
	return orm.UserID
}

// GetOrderNumber returns the order number
func (orm *OrderReadModel) GetOrderNumber() string {
	return orm.OrderNumber
}

// GetTotalAmount returns the total amount
func (orm *OrderReadModel) GetTotalAmount() float64 {
	return orm.TotalAmount
}

// GetCurrency returns the currency
func (orm *OrderReadModel) GetCurrency() string {
	return orm.Currency
}

// GetStatus returns the order status
func (orm *OrderReadModel) GetStatus() string {
	return orm.Status
}

// GetOrderDate returns the order date
func (orm *OrderReadModel) GetOrderDate() time.Time {
	return orm.OrderDate
}

// GetItems returns the order items
func (orm *OrderReadModel) GetItems() []OrderItem {
	return orm.Items
}

// PaymentReadModel represents a payment read model
type PaymentReadModel struct {
	*ConcreteReadModel
	PaymentID     string    `json:"payment_id"`
	OrderID       string    `json:"order_id"`
	UserID        string    `json:"user_id"`
	Amount        float64   `json:"amount"`
	Currency      string    `json:"currency"`
	PaymentMethod string    `json:"payment_method"`
	Status        string    `json:"status"`
	TransactionID string    `json:"transaction_id"`
	Gateway       string    `json:"gateway"`
	PaymentDate   time.Time `json:"payment_date"`
}

// NewPaymentReadModel creates a new payment read model
func NewPaymentReadModel(paymentID, orderID, userID string, amount float64, currency, paymentMethod, status, transactionID, gateway string, paymentDate time.Time) *PaymentReadModel {
	return &PaymentReadModel{
		ConcreteReadModel: &ConcreteReadModel{
			ID:        generateID(),
			Type:      "payment",
			Data:      make(map[string]interface{}),
			Metadata:  make(map[string]interface{}),
			Version:   1,
			Timestamp: time.Now(),
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
			Active:    true,
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
		PaymentDate:   paymentDate,
	}
}

// GetPaymentID returns the payment ID
func (prm *PaymentReadModel) GetPaymentID() string {
	return prm.PaymentID
}

// GetOrderID returns the order ID
func (prm *PaymentReadModel) GetOrderID() string {
	return prm.OrderID
}

// GetUserID returns the user ID
func (prm *PaymentReadModel) GetUserID() string {
	return prm.UserID
}

// GetAmount returns the payment amount
func (prm *PaymentReadModel) GetAmount() float64 {
	return prm.Amount
}

// GetCurrency returns the currency
func (prm *PaymentReadModel) GetCurrency() string {
	return prm.Currency
}

// GetPaymentMethod returns the payment method
func (prm *PaymentReadModel) GetPaymentMethod() string {
	return prm.PaymentMethod
}

// GetStatus returns the payment status
func (prm *PaymentReadModel) GetStatus() string {
	return prm.Status
}

// GetTransactionID returns the transaction ID
func (prm *PaymentReadModel) GetTransactionID() string {
	return prm.TransactionID
}

// GetGateway returns the payment gateway
func (prm *PaymentReadModel) GetGateway() string {
	return prm.Gateway
}

// GetPaymentDate returns the payment date
func (prm *PaymentReadModel) GetPaymentDate() time.Time {
	return prm.PaymentDate
}
