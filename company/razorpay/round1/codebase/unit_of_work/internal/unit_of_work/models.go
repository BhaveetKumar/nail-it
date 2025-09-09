package unit_of_work

import (
	"time"
)

// UserEntity represents a user entity
type UserEntity struct {
	*ConcreteEntity
	Email     string `json:"email"`
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
	Role      string `json:"role"`
	Status    string `json:"status"`
}

// NewUserEntity creates a new user entity
func NewUserEntity(name, description, email, firstName, lastName, role string) *UserEntity {
	return &UserEntity{
		ConcreteEntity: &ConcreteEntity{
			ID:          generateID(),
			Type:        "user",
			Name:        name,
			Description: description,
			Metadata:    make(map[string]interface{}),
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
			Active:      true,
			Dirty:       false,
			New:         true,
			Deleted:     false,
		},
		Email:     email,
		FirstName: firstName,
		LastName:  lastName,
		Role:      role,
		Status:    "active",
	}
}

// GetEmail returns the user email
func (ue *UserEntity) GetEmail() string {
	return ue.Email
}

// SetEmail sets the user email
func (ue *UserEntity) SetEmail(email string) {
	ue.Email = email
	ue.UpdatedAt = time.Now()
	ue.Dirty = true
}

// GetFirstName returns the user first name
func (ue *UserEntity) GetFirstName() string {
	return ue.FirstName
}

// SetFirstName sets the user first name
func (ue *UserEntity) SetFirstName(firstName string) {
	ue.FirstName = firstName
	ue.UpdatedAt = time.Now()
	ue.Dirty = true
}

// GetLastName returns the user last name
func (ue *UserEntity) GetLastName() string {
	return ue.LastName
}

// SetLastName sets the user last name
func (ue *UserEntity) SetLastName(lastName string) {
	ue.LastName = lastName
	ue.UpdatedAt = time.Now()
	ue.Dirty = true
}

// GetRole returns the user role
func (ue *UserEntity) GetRole() string {
	return ue.Role
}

// SetRole sets the user role
func (ue *UserEntity) SetRole(role string) {
	ue.Role = role
	ue.UpdatedAt = time.Now()
	ue.Dirty = true
}

// GetStatus returns the user status
func (ue *UserEntity) GetStatus() string {
	return ue.Status
}

// SetStatus sets the user status
func (ue *UserEntity) SetStatus(status string) {
	ue.Status = status
	ue.UpdatedAt = time.Now()
	ue.Dirty = true
}

// OrderEntity represents an order entity
type OrderEntity struct {
	*ConcreteEntity
	UserID      string    `json:"user_id"`
	OrderNumber string    `json:"order_number"`
	TotalAmount float64   `json:"total_amount"`
	Currency    string    `json:"currency"`
	Status      string    `json:"status"`
	OrderDate   time.Time `json:"order_date"`
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

// NewOrderEntity creates a new order entity
func NewOrderEntity(name, description, userID, orderNumber string, totalAmount float64, currency string) *OrderEntity {
	return &OrderEntity{
		ConcreteEntity: &ConcreteEntity{
			ID:          generateID(),
			Type:        "order",
			Name:        name,
			Description: description,
			Metadata:    make(map[string]interface{}),
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
			Active:      true,
			Dirty:       false,
			New:         true,
			Deleted:     false,
		},
		UserID:      userID,
		OrderNumber: orderNumber,
		TotalAmount: totalAmount,
		Currency:    currency,
		Status:      "pending",
		OrderDate:   time.Now(),
		Items:       make([]OrderItem, 0),
	}
}

// GetUserID returns the order user ID
func (oe *OrderEntity) GetUserID() string {
	return oe.UserID
}

// SetUserID sets the order user ID
func (oe *OrderEntity) SetUserID(userID string) {
	oe.UserID = userID
	oe.UpdatedAt = time.Now()
	oe.Dirty = true
}

// GetOrderNumber returns the order number
func (oe *OrderEntity) GetOrderNumber() string {
	return oe.OrderNumber
}

// SetOrderNumber sets the order number
func (oe *OrderEntity) SetOrderNumber(orderNumber string) {
	oe.OrderNumber = orderNumber
	oe.UpdatedAt = time.Now()
	oe.Dirty = true
}

// GetTotalAmount returns the order total amount
func (oe *OrderEntity) GetTotalAmount() float64 {
	return oe.TotalAmount
}

// SetTotalAmount sets the order total amount
func (oe *OrderEntity) SetTotalAmount(totalAmount float64) {
	oe.TotalAmount = totalAmount
	oe.UpdatedAt = time.Now()
	oe.Dirty = true
}

// GetCurrency returns the order currency
func (oe *OrderEntity) GetCurrency() string {
	return oe.Currency
}

// SetCurrency sets the order currency
func (oe *OrderEntity) SetCurrency(currency string) {
	oe.Currency = currency
	oe.UpdatedAt = time.Now()
	oe.Dirty = true
}

// GetStatus returns the order status
func (oe *OrderEntity) GetStatus() string {
	return oe.Status
}

// SetStatus sets the order status
func (oe *OrderEntity) SetStatus(status string) {
	oe.Status = status
	oe.UpdatedAt = time.Now()
	oe.Dirty = true
}

// GetOrderDate returns the order date
func (oe *OrderEntity) GetOrderDate() time.Time {
	return oe.OrderDate
}

// SetOrderDate sets the order date
func (oe *OrderEntity) SetOrderDate(orderDate time.Time) {
	oe.OrderDate = orderDate
	oe.UpdatedAt = time.Now()
	oe.Dirty = true
}

// GetItems returns the order items
func (oe *OrderEntity) GetItems() []OrderItem {
	return oe.Items
}

// SetItems sets the order items
func (oe *OrderEntity) SetItems(items []OrderItem) {
	oe.Items = items
	oe.UpdatedAt = time.Now()
	oe.Dirty = true
}

// AddItem adds an item to the order
func (oe *OrderEntity) AddItem(item OrderItem) {
	oe.Items = append(oe.Items, item)
	oe.UpdatedAt = time.Now()
	oe.Dirty = true
}

// RemoveItem removes an item from the order
func (oe *OrderEntity) RemoveItem(itemID string) {
	for i, item := range oe.Items {
		if item.ID == itemID {
			oe.Items = append(oe.Items[:i], oe.Items[i+1:]...)
			break
		}
	}
	oe.UpdatedAt = time.Now()
	oe.Dirty = true
}

// ProductEntity represents a product entity
type ProductEntity struct {
	*ConcreteEntity
	SKU         string  `json:"sku"`
	Price       float64 `json:"price"`
	Currency    string  `json:"currency"`
	Category    string  `json:"category"`
	Brand       string  `json:"brand"`
	Description string  `json:"description"`
	Stock       int     `json:"stock"`
	Status      string  `json:"status"`
}

// NewProductEntity creates a new product entity
func NewProductEntity(name, description, sku string, price float64, currency, category, brand string) *ProductEntity {
	return &ProductEntity{
		ConcreteEntity: &ConcreteEntity{
			ID:          generateID(),
			Type:        "product",
			Name:        name,
			Description: description,
			Metadata:    make(map[string]interface{}),
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
			Active:      true,
			Dirty:       false,
			New:         true,
			Deleted:     false,
		},
		SKU:         sku,
		Price:       price,
		Currency:    currency,
		Category:    category,
		Brand:       brand,
		Description: description,
		Stock:       0,
		Status:      "active",
	}
}

// GetSKU returns the product SKU
func (pe *ProductEntity) GetSKU() string {
	return pe.SKU
}

// SetSKU sets the product SKU
func (pe *ProductEntity) SetSKU(sku string) {
	pe.SKU = sku
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetPrice returns the product price
func (pe *ProductEntity) GetPrice() float64 {
	return pe.Price
}

// SetPrice sets the product price
func (pe *ProductEntity) SetPrice(price float64) {
	pe.Price = price
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetCurrency returns the product currency
func (pe *ProductEntity) GetCurrency() string {
	return pe.Currency
}

// SetCurrency sets the product currency
func (pe *ProductEntity) SetCurrency(currency string) {
	pe.Currency = currency
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetCategory returns the product category
func (pe *ProductEntity) GetCategory() string {
	return pe.Category
}

// SetCategory sets the product category
func (pe *ProductEntity) SetCategory(category string) {
	pe.Category = category
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetBrand returns the product brand
func (pe *ProductEntity) GetBrand() string {
	return pe.Brand
}

// SetBrand sets the product brand
func (pe *ProductEntity) SetBrand(brand string) {
	pe.Brand = brand
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetStock returns the product stock
func (pe *ProductEntity) GetStock() int {
	return pe.Stock
}

// SetStock sets the product stock
func (pe *ProductEntity) SetStock(stock int) {
	pe.Stock = stock
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetStatus returns the product status
func (pe *ProductEntity) GetStatus() string {
	return pe.Status
}

// SetStatus sets the product status
func (pe *ProductEntity) SetStatus(status string) {
	pe.Status = status
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// PaymentEntity represents a payment entity
type PaymentEntity struct {
	*ConcreteEntity
	OrderID       string    `json:"order_id"`
	UserID        string    `json:"user_id"`
	Amount        float64   `json:"amount"`
	Currency      string    `json:"currency"`
	PaymentMethod string    `json:"payment_method"`
	Status        string    `json:"status"`
	TransactionID string    `json:"transaction_id"`
	PaymentDate   time.Time `json:"payment_date"`
	Gateway       string    `json:"gateway"`
}

// NewPaymentEntity creates a new payment entity
func NewPaymentEntity(name, description, orderID, userID string, amount float64, currency, paymentMethod string) *PaymentEntity {
	return &PaymentEntity{
		ConcreteEntity: &ConcreteEntity{
			ID:          generateID(),
			Type:        "payment",
			Name:        name,
			Description: description,
			Metadata:    make(map[string]interface{}),
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
			Active:      true,
			Dirty:       false,
			New:         true,
			Deleted:     false,
		},
		OrderID:       orderID,
		UserID:        userID,
		Amount:        amount,
		Currency:      currency,
		PaymentMethod: paymentMethod,
		Status:        "pending",
		TransactionID: "",
		PaymentDate:   time.Time{},
		Gateway:       "",
	}
}

// GetOrderID returns the payment order ID
func (pe *PaymentEntity) GetOrderID() string {
	return pe.OrderID
}

// SetOrderID sets the payment order ID
func (pe *PaymentEntity) SetOrderID(orderID string) {
	pe.OrderID = orderID
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetUserID returns the payment user ID
func (pe *PaymentEntity) GetUserID() string {
	return pe.UserID
}

// SetUserID sets the payment user ID
func (pe *PaymentEntity) SetUserID(userID string) {
	pe.UserID = userID
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetAmount returns the payment amount
func (pe *PaymentEntity) GetAmount() float64 {
	return pe.Amount
}

// SetAmount sets the payment amount
func (pe *PaymentEntity) SetAmount(amount float64) {
	pe.Amount = amount
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetCurrency returns the payment currency
func (pe *PaymentEntity) GetCurrency() string {
	return pe.Currency
}

// SetCurrency sets the payment currency
func (pe *PaymentEntity) SetCurrency(currency string) {
	pe.Currency = currency
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetPaymentMethod returns the payment method
func (pe *PaymentEntity) GetPaymentMethod() string {
	return pe.PaymentMethod
}

// SetPaymentMethod sets the payment method
func (pe *PaymentEntity) SetPaymentMethod(paymentMethod string) {
	pe.PaymentMethod = paymentMethod
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetStatus returns the payment status
func (pe *PaymentEntity) GetStatus() string {
	return pe.Status
}

// SetStatus sets the payment status
func (pe *PaymentEntity) SetStatus(status string) {
	pe.Status = status
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetTransactionID returns the payment transaction ID
func (pe *PaymentEntity) GetTransactionID() string {
	return pe.TransactionID
}

// SetTransactionID sets the payment transaction ID
func (pe *PaymentEntity) SetTransactionID(transactionID string) {
	pe.TransactionID = transactionID
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetPaymentDate returns the payment date
func (pe *PaymentEntity) GetPaymentDate() time.Time {
	return pe.PaymentDate
}

// SetPaymentDate sets the payment date
func (pe *PaymentEntity) SetPaymentDate(paymentDate time.Time) {
	pe.PaymentDate = paymentDate
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// GetGateway returns the payment gateway
func (pe *PaymentEntity) GetGateway() string {
	return pe.Gateway
}

// SetGateway sets the payment gateway
func (pe *PaymentEntity) SetGateway(gateway string) {
	pe.Gateway = gateway
	pe.UpdatedAt = time.Now()
	pe.Dirty = true
}

// UserRepository represents a user repository
type UserRepository struct {
	*ConcreteRepository
}

// NewUserRepository creates a new user repository
func NewUserRepository(name, description string) *UserRepository {
	return &UserRepository{
		ConcreteRepository: &ConcreteRepository{
			ID:          generateID(),
			Type:        "user",
			Name:        name,
			Description: description,
			Metadata:    make(map[string]interface{}),
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
			Active:      true,
			Entities:    make(map[string]Entity),
		},
	}
}

// OrderRepository represents an order repository
type OrderRepository struct {
	*ConcreteRepository
}

// NewOrderRepository creates a new order repository
func NewOrderRepository(name, description string) *OrderRepository {
	return &OrderRepository{
		ConcreteRepository: &ConcreteRepository{
			ID:          generateID(),
			Type:        "order",
			Name:        name,
			Description: description,
			Metadata:    make(map[string]interface{}),
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
			Active:      true,
			Entities:    make(map[string]Entity),
		},
	}
}

// ProductRepository represents a product repository
type ProductRepository struct {
	*ConcreteRepository
}

// NewProductRepository creates a new product repository
func NewProductRepository(name, description string) *ProductRepository {
	return &ProductRepository{
		ConcreteRepository: &ConcreteRepository{
			ID:          generateID(),
			Type:        "product",
			Name:        name,
			Description: description,
			Metadata:    make(map[string]interface{}),
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
			Active:      true,
			Entities:    make(map[string]Entity),
		},
	}
}

// PaymentRepository represents a payment repository
type PaymentRepository struct {
	*ConcreteRepository
}

// NewPaymentRepository creates a new payment repository
func NewPaymentRepository(name, description string) *PaymentRepository {
	return &PaymentRepository{
		ConcreteRepository: &ConcreteRepository{
			ID:          generateID(),
			Type:        "payment",
			Name:        name,
			Description: description,
			Metadata:    make(map[string]interface{}),
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
			Active:      true,
			Entities:    make(map[string]Entity),
		},
	}
}
