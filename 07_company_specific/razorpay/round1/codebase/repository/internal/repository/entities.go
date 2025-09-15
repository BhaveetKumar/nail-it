package repository

import (
	"time"
)

// User represents a user entity
type User struct {
	ID        string    `json:"id" bson:"id" db:"id"`
	Email     string    `json:"email" bson:"email" db:"email"`
	Name      string    `json:"name" bson:"name" db:"name"`
	Status    string    `json:"status" bson:"status" db:"status"`
	CreatedAt time.Time `json:"created_at" bson:"created_at" db:"created_at"`
	UpdatedAt time.Time `json:"updated_at" bson:"updated_at" db:"updated_at"`
	DeletedAt *time.Time `json:"deleted_at,omitempty" bson:"deleted_at,omitempty" db:"deleted_at"`
}

func (u *User) GetID() string {
	return u.ID
}

func (u *User) GetCreatedAt() time.Time {
	return u.CreatedAt
}

func (u *User) GetUpdatedAt() time.Time {
	return u.UpdatedAt
}

// Payment represents a payment entity
type Payment struct {
	ID            string                 `json:"id" bson:"id" db:"id"`
	UserID        string                 `json:"user_id" bson:"user_id" db:"user_id"`
	Amount        float64                `json:"amount" bson:"amount" db:"amount"`
	Currency      string                 `json:"currency" bson:"currency" db:"currency"`
	Status        string                 `json:"status" bson:"status" db:"status"`
	Gateway       string                 `json:"gateway" bson:"gateway" db:"gateway"`
	TransactionID string                 `json:"transaction_id" bson:"transaction_id" db:"transaction_id"`
	Metadata      map[string]interface{} `json:"metadata" bson:"metadata" db:"metadata"`
	CreatedAt     time.Time              `json:"created_at" bson:"created_at" db:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at" bson:"updated_at" db:"updated_at"`
	DeletedAt     *time.Time             `json:"deleted_at,omitempty" bson:"deleted_at,omitempty" db:"deleted_at"`
}

func (p *Payment) GetID() string {
	return p.ID
}

func (p *Payment) GetCreatedAt() time.Time {
	return p.CreatedAt
}

func (p *Payment) GetUpdatedAt() time.Time {
	return p.UpdatedAt
}

// Order represents an order entity
type Order struct {
	ID          string                 `json:"id" bson:"id" db:"id"`
	UserID      string                 `json:"user_id" bson:"user_id" db:"user_id"`
	PaymentID   string                 `json:"payment_id" bson:"payment_id" db:"payment_id"`
	TotalAmount float64                `json:"total_amount" bson:"total_amount" db:"total_amount"`
	Currency    string                 `json:"currency" bson:"currency" db:"currency"`
	Status      string                 `json:"status" bson:"status" db:"status"`
	Items       []OrderItem            `json:"items" bson:"items" db:"items"`
	Metadata    map[string]interface{} `json:"metadata" bson:"metadata" db:"metadata"`
	CreatedAt   time.Time              `json:"created_at" bson:"created_at" db:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" bson:"updated_at" db:"updated_at"`
	DeletedAt   *time.Time             `json:"deleted_at,omitempty" bson:"deleted_at,omitempty" db:"deleted_at"`
}

func (o *Order) GetID() string {
	return o.ID
}

func (o *Order) GetCreatedAt() time.Time {
	return o.CreatedAt
}

func (o *Order) GetUpdatedAt() time.Time {
	return o.UpdatedAt
}

// OrderItem represents an order item
type OrderItem struct {
	ID        string  `json:"id" bson:"id" db:"id"`
	ProductID string  `json:"product_id" bson:"product_id" db:"product_id"`
	Quantity  int     `json:"quantity" bson:"quantity" db:"quantity"`
	Price     float64 `json:"price" bson:"price" db:"price"`
	Total     float64 `json:"total" bson:"total" db:"total"`
}

// Product represents a product entity
type Product struct {
	ID          string                 `json:"id" bson:"id" db:"id"`
	Name        string                 `json:"name" bson:"name" db:"name"`
	Description string                 `json:"description" bson:"description" db:"description"`
	Price       float64                `json:"price" bson:"price" db:"price"`
	Currency    string                 `json:"currency" bson:"currency" db:"currency"`
	Category    string                 `json:"category" bson:"category" db:"category"`
	Stock       int                    `json:"stock" bson:"stock" db:"stock"`
	Status      string                 `json:"status" bson:"status" db:"status"`
	Metadata    map[string]interface{} `json:"metadata" bson:"metadata" db:"metadata"`
	CreatedAt   time.Time              `json:"created_at" bson:"created_at" db:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" bson:"updated_at" db:"updated_at"`
	DeletedAt   *time.Time             `json:"deleted_at,omitempty" bson:"deleted_at,omitempty" db:"deleted_at"`
}

func (p *Product) GetID() string {
	return p.ID
}

func (p *Product) GetCreatedAt() time.Time {
	return p.CreatedAt
}

func (p *Product) GetUpdatedAt() time.Time {
	return p.UpdatedAt
}

// Specifications for common queries

// UserByEmailSpecification finds users by email
type UserByEmailSpecification struct {
	Email string
}

func (s *UserByEmailSpecification) IsSatisfiedBy(entity Entity) bool {
	user, ok := entity.(*User)
	if !ok {
		return false
	}
	return user.Email == s.Email
}

func (s *UserByEmailSpecification) ToSQL() (string, []interface{}) {
	return "email = ?", []interface{}{s.Email}
}

func (s *UserByEmailSpecification) ToMongoFilter() map[string]interface{} {
	return map[string]interface{}{"email": s.Email}
}

// UserByStatusSpecification finds users by status
type UserByStatusSpecification struct {
	Status string
}

func (s *UserByStatusSpecification) IsSatisfiedBy(entity Entity) bool {
	user, ok := entity.(*User)
	if !ok {
		return false
	}
	return user.Status == s.Status
}

func (s *UserByStatusSpecification) ToSQL() (string, []interface{}) {
	return "status = ?", []interface{}{s.Status}
}

func (s *UserByStatusSpecification) ToMongoFilter() map[string]interface{} {
	return map[string]interface{}{"status": s.Status}
}

// PaymentByUserIDSpecification finds payments by user ID
type PaymentByUserIDSpecification struct {
	UserID string
}

func (s *PaymentByUserIDSpecification) IsSatisfiedBy(entity Entity) bool {
	payment, ok := entity.(*Payment)
	if !ok {
		return false
	}
	return payment.UserID == s.UserID
}

func (s *PaymentByUserIDSpecification) ToSQL() (string, []interface{}) {
	return "user_id = ?", []interface{}{s.UserID}
}

func (s *PaymentByUserIDSpecification) ToMongoFilter() map[string]interface{} {
	return map[string]interface{}{"user_id": s.UserID}
}

// PaymentByStatusSpecification finds payments by status
type PaymentByStatusSpecification struct {
	Status string
}

func (s *PaymentByStatusSpecification) IsSatisfiedBy(entity Entity) bool {
	payment, ok := entity.(*Payment)
	if !ok {
		return false
	}
	return payment.Status == s.Status
}

func (s *PaymentByStatusSpecification) ToSQL() (string, []interface{}) {
	return "status = ?", []interface{}{s.Status}
}

func (s *PaymentByStatusSpecification) ToMongoFilter() map[string]interface{} {
	return map[string]interface{}{"status": s.Status}
}

// PaymentByGatewaySpecification finds payments by gateway
type PaymentByGatewaySpecification struct {
	Gateway string
}

func (s *PaymentByGatewaySpecification) IsSatisfiedBy(entity Entity) bool {
	payment, ok := entity.(*Payment)
	if !ok {
		return false
	}
	return payment.Gateway == s.Gateway
}

func (s *PaymentByGatewaySpecification) ToSQL() (string, []interface{}) {
	return "gateway = ?", []interface{}{s.Gateway}
}

func (s *PaymentByGatewaySpecification) ToMongoFilter() map[string]interface{} {
	return map[string]interface{}{"gateway": s.Gateway}
}

// PaymentByAmountRangeSpecification finds payments within amount range
type PaymentByAmountRangeSpecification struct {
	MinAmount float64
	MaxAmount float64
}

func (s *PaymentByAmountRangeSpecification) IsSatisfiedBy(entity Entity) bool {
	payment, ok := entity.(*Payment)
	if !ok {
		return false
	}
	return payment.Amount >= s.MinAmount && payment.Amount <= s.MaxAmount
}

func (s *PaymentByAmountRangeSpecification) ToSQL() (string, []interface{}) {
	return "amount >= ? AND amount <= ?", []interface{}{s.MinAmount, s.MaxAmount}
}

func (s *PaymentByAmountRangeSpecification) ToMongoFilter() map[string]interface{} {
	return map[string]interface{}{
		"amount": map[string]interface{}{
			"$gte": s.MinAmount,
			"$lte": s.MaxAmount,
		},
	}
}

// OrderByUserIDSpecification finds orders by user ID
type OrderByUserIDSpecification struct {
	UserID string
}

func (s *OrderByUserIDSpecification) IsSatisfiedBy(entity Entity) bool {
	order, ok := entity.(*Order)
	if !ok {
		return false
	}
	return order.UserID == s.UserID
}

func (s *OrderByUserIDSpecification) ToSQL() (string, []interface{}) {
	return "user_id = ?", []interface{}{s.UserID}
}

func (s *OrderByUserIDSpecification) ToMongoFilter() map[string]interface{} {
	return map[string]interface{}{"user_id": s.UserID}
}

// OrderByStatusSpecification finds orders by status
type OrderByStatusSpecification struct {
	Status string
}

func (s *OrderByStatusSpecification) IsSatisfiedBy(entity Entity) bool {
	order, ok := entity.(*Order)
	if !ok {
		return false
	}
	return order.Status == s.Status
}

func (s *OrderByStatusSpecification) ToSQL() (string, []interface{}) {
	return "status = ?", []interface{}{s.Status}
}

func (s *OrderByStatusSpecification) ToMongoFilter() map[string]interface{} {
	return map[string]interface{}{"status": s.Status}
}

// ProductByCategorySpecification finds products by category
type ProductByCategorySpecification struct {
	Category string
}

func (s *ProductByCategorySpecification) IsSatisfiedBy(entity Entity) bool {
	product, ok := entity.(*Product)
	if !ok {
		return false
	}
	return product.Category == s.Category
}

func (s *ProductByCategorySpecification) ToSQL() (string, []interface{}) {
	return "category = ?", []interface{}{s.Category}
}

func (s *ProductByCategorySpecification) ToMongoFilter() map[string]interface{} {
	return map[string]interface{}{"category": s.Category}
}

// ProductByPriceRangeSpecification finds products within price range
type ProductByPriceRangeSpecification struct {
	MinPrice float64
	MaxPrice float64
}

func (s *ProductByPriceRangeSpecification) IsSatisfiedBy(entity Entity) bool {
	product, ok := entity.(*Product)
	if !ok {
		return false
	}
	return product.Price >= s.MinPrice && product.Price <= s.MaxPrice
}

func (s *ProductByPriceRangeSpecification) ToSQL() (string, []interface{}) {
	return "price >= ? AND price <= ?", []interface{}{s.MinPrice, s.MaxPrice}
}

func (s *ProductByPriceRangeSpecification) ToMongoFilter() map[string]interface{} {
	return map[string]interface{}{
		"price": map[string]interface{}{
			"$gte": s.MinPrice,
			"$lte": s.MaxPrice,
		},
	}
}

// ProductInStockSpecification finds products that are in stock
type ProductInStockSpecification struct{}

func (s *ProductInStockSpecification) IsSatisfiedBy(entity Entity) bool {
	product, ok := entity.(*Product)
	if !ok {
		return false
	}
	return product.Stock > 0
}

func (s *ProductInStockSpecification) ToSQL() (string, []interface{}) {
	return "stock > 0", []interface{}{}
}

func (s *ProductInStockSpecification) ToMongoFilter() map[string]interface{} {
	return map[string]interface{}{"stock": map[string]interface{}{"$gt": 0}}
}

// DateRangeSpecification finds entities within date range
type DateRangeSpecification struct {
	StartDate time.Time
	EndDate   time.Time
	Field     string // "created_at", "updated_at", etc.
}

func (s *DateRangeSpecification) IsSatisfiedBy(entity Entity) bool {
	var fieldTime time.Time
	
	switch s.Field {
	case "created_at":
		fieldTime = entity.GetCreatedAt()
	case "updated_at":
		fieldTime = entity.GetUpdatedAt()
	default:
		return false
	}
	
	return fieldTime.After(s.StartDate) && fieldTime.Before(s.EndDate)
}

func (s *DateRangeSpecification) ToSQL() (string, []interface{}) {
	return s.Field + " >= ? AND " + s.Field + " <= ?", []interface{}{s.StartDate, s.EndDate}
}

func (s *DateRangeSpecification) ToMongoFilter() map[string]interface{} {
	return map[string]interface{}{
		s.Field: map[string]interface{}{
			"$gte": s.StartDate,
			"$lte": s.EndDate,
		},
	}
}
