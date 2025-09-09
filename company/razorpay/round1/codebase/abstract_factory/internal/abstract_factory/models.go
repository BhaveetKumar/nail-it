package abstract_factory

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Common errors
var (
	ErrFactoryNotFound      = errors.New("factory not found")
	ErrFactoryAlreadyExists = errors.New("factory already exists")
	ErrFactoryInactive      = errors.New("factory is inactive")
	ErrProductNotFound      = errors.New("product not found")
	ErrProductAlreadyExists = errors.New("product already exists")
	ErrProductInactive      = errors.New("product is inactive")
	ErrInvalidFactoryType   = errors.New("invalid factory type")
	ErrInvalidProductType   = errors.New("invalid product type")
	ErrInvalidConfiguration = errors.New("invalid configuration")
	ErrValidationFailed     = errors.New("validation failed")
	ErrServiceInactive      = errors.New("service is inactive")
	ErrRegistryNotFound     = errors.New("registry not found")
	ErrRegistryInactive     = errors.New("registry is inactive")
)

// BaseProduct represents a base product implementation
type BaseProduct struct {
	ID          string                 `json:"id" yaml:"id"`
	Name        string                 `json:"name" yaml:"name"`
	Type        string                 `json:"type" yaml:"type"`
	Description string                 `json:"description" yaml:"description"`
	Price       float64                `json:"price" yaml:"price"`
	Metadata    map[string]interface{} `json:"metadata" yaml:"metadata"`
	Active      bool                   `json:"active" yaml:"active"`
	CreatedAt   time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" yaml:"updated_at"`
}

// GetID returns the product ID
func (p *BaseProduct) GetID() string {
	return p.ID
}

// GetName returns the product name
func (p *BaseProduct) GetName() string {
	return p.Name
}

// GetType returns the product type
func (p *BaseProduct) GetType() string {
	return p.Type
}

// GetDescription returns the product description
func (p *BaseProduct) GetDescription() string {
	return p.Description
}

// GetPrice returns the product price
func (p *BaseProduct) GetPrice() float64 {
	return p.Price
}

// GetCreatedAt returns the creation time
func (p *BaseProduct) GetCreatedAt() time.Time {
	return p.CreatedAt
}

// GetUpdatedAt returns the last update time
func (p *BaseProduct) GetUpdatedAt() time.Time {
	return p.UpdatedAt
}

// GetMetadata returns the product metadata
func (p *BaseProduct) GetMetadata() map[string]interface{} {
	return p.Metadata
}

// SetMetadata sets the product metadata
func (p *BaseProduct) SetMetadata(key string, value interface{}) {
	if p.Metadata == nil {
		p.Metadata = make(map[string]interface{})
	}
	p.Metadata[key] = value
	p.UpdatedAt = time.Now()
}

// IsActive returns whether the product is active
func (p *BaseProduct) IsActive() bool {
	return p.Active
}

// SetActive sets the active status
func (p *BaseProduct) SetActive(active bool) {
	p.Active = active
	p.UpdatedAt = time.Now()
}

// Validate validates the product
func (p *BaseProduct) Validate() error {
	if p.ID == "" {
		return fmt.Errorf("product ID is required")
	}
	if p.Name == "" {
		return fmt.Errorf("product name is required")
	}
	if p.Type == "" {
		return fmt.Errorf("product type is required")
	}
	if p.Price < 0 {
		return fmt.Errorf("product price cannot be negative")
	}
	return nil
}

// GetStats returns product statistics
func (p *BaseProduct) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"id":          p.ID,
		"name":        p.Name,
		"type":        p.Type,
		"description": p.Description,
		"price":       p.Price,
		"active":      p.Active,
		"created_at":  p.CreatedAt,
		"updated_at":  p.UpdatedAt,
		"metadata":    p.Metadata,
	}
}

// ConcreteProductA represents a concrete implementation of ProductA
type ConcreteProductA struct {
	BaseProduct
	Category    string `json:"category" yaml:"category"`
	SubCategory string `json:"sub_category" yaml:"sub_category"`
}

// NewConcreteProductA creates a new ConcreteProductA
func NewConcreteProductA(id, name, description string, price float64) *ConcreteProductA {
	return &ConcreteProductA{
		BaseProduct: BaseProduct{
			ID:          id,
			Name:        name,
			Type:        "ProductA",
			Description: description,
			Price:       price,
			Metadata:    make(map[string]interface{}),
			Active:      true,
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		},
		Category:    "CategoryA",
		SubCategory: "SubCategoryA",
	}
}

// Process processes the product
func (p *ConcreteProductA) Process(ctx context.Context) error {
	if !p.Active {
		return ErrProductInactive
	}
	// Simulate processing
	time.Sleep(100 * time.Millisecond)
	p.UpdatedAt = time.Now()
	return nil
}

// GetStats returns product statistics
func (p *ConcreteProductA) GetStats() map[string]interface{} {
	stats := p.BaseProduct.GetStats()
	stats["category"] = p.Category
	stats["sub_category"] = p.SubCategory
	return stats
}

// ConcreteProductB represents a concrete implementation of ProductB
type ConcreteProductB struct {
	BaseProduct
	Brand string `json:"brand" yaml:"brand"`
	Model string `json:"model" yaml:"model"`
}

// NewConcreteProductB creates a new ConcreteProductB
func NewConcreteProductB(id, name, description string, price float64) *ConcreteProductB {
	return &ConcreteProductB{
		BaseProduct: BaseProduct{
			ID:          id,
			Name:        name,
			Type:        "ProductB",
			Description: description,
			Price:       price,
			Metadata:    make(map[string]interface{}),
			Active:      true,
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		},
		Brand: "BrandB",
		Model: "ModelB",
	}
}

// Process processes the product
func (p *ConcreteProductB) Process(ctx context.Context) error {
	if !p.Active {
		return ErrProductInactive
	}
	// Simulate processing
	time.Sleep(150 * time.Millisecond)
	p.UpdatedAt = time.Now()
	return nil
}

// GetStats returns product statistics
func (p *ConcreteProductB) GetStats() map[string]interface{} {
	stats := p.BaseProduct.GetStats()
	stats["brand"] = p.Brand
	stats["model"] = p.Model
	return stats
}

// ConcreteProductC represents a concrete implementation of ProductC
type ConcreteProductC struct {
	BaseProduct
	Manufacturer string `json:"manufacturer" yaml:"manufacturer"`
	SerialNumber string `json:"serial_number" yaml:"serial_number"`
}

// NewConcreteProductC creates a new ConcreteProductC
func NewConcreteProductC(id, name, description string, price float64) *ConcreteProductC {
	return &ConcreteProductC{
		BaseProduct: BaseProduct{
			ID:          id,
			Name:        name,
			Type:        "ProductC",
			Description: description,
			Price:       price,
			Metadata:    make(map[string]interface{}),
			Active:      true,
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		},
		Manufacturer: "ManufacturerC",
		SerialNumber: "SNC-001",
	}
}

// Process processes the product
func (p *ConcreteProductC) Process(ctx context.Context) error {
	if !p.Active {
		return ErrProductInactive
	}
	// Simulate processing
	time.Sleep(200 * time.Millisecond)
	p.UpdatedAt = time.Now()
	return nil
}

// GetStats returns product statistics
func (p *ConcreteProductC) GetStats() map[string]interface{} {
	stats := p.BaseProduct.GetStats()
	stats["manufacturer"] = p.Manufacturer
	stats["serial_number"] = p.SerialNumber
	return stats
}

// ConcreteFactory1 represents a concrete implementation of AbstractFactory
type ConcreteFactory1 struct {
	config    *FactoryConfig
	createdAt time.Time
	updatedAt time.Time
	active    bool
}

// NewConcreteFactory1 creates a new ConcreteFactory1
func NewConcreteFactory1(config *FactoryConfig) *ConcreteFactory1 {
	return &ConcreteFactory1{
		config:    config,
		createdAt: time.Now(),
		updatedAt: time.Now(),
		active:    true,
	}
}

// CreateProductA creates a ProductA
func (f *ConcreteFactory1) CreateProductA() ProductA {
	return NewConcreteProductA("product-a-1", "Product A1", "Description for Product A1", 100.0)
}

// CreateProductB creates a ProductB
func (f *ConcreteFactory1) CreateProductB() ProductB {
	return NewConcreteProductB("product-b-1", "Product B1", "Description for Product B1", 200.0)
}

// CreateProductC creates a ProductC
func (f *ConcreteFactory1) CreateProductC() ProductC {
	return NewConcreteProductC("product-c-1", "Product C1", "Description for Product C1", 300.0)
}

// GetFactoryType returns the factory type
func (f *ConcreteFactory1) GetFactoryType() string {
	return "ConcreteFactory1"
}

// GetFactoryInfo returns factory information
func (f *ConcreteFactory1) GetFactoryInfo() map[string]interface{} {
	return map[string]interface{}{
		"type":        f.GetFactoryType(),
		"name":        f.config.Name,
		"description": f.config.Description,
		"version":     f.config.Version,
		"active":      f.active,
		"created_at":  f.createdAt,
		"updated_at":  f.updatedAt,
		"metadata":    f.config.Metadata,
	}
}

// IsActive returns whether the factory is active
func (f *ConcreteFactory1) IsActive() bool {
	return f.active
}

// SetActive sets the active status
func (f *ConcreteFactory1) SetActive(active bool) {
	f.active = active
	f.updatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (f *ConcreteFactory1) GetCreatedAt() time.Time {
	return f.createdAt
}

// GetUpdatedAt returns the last update time
func (f *ConcreteFactory1) GetUpdatedAt() time.Time {
	return f.updatedAt
}

// GetMetadata returns the factory metadata
func (f *ConcreteFactory1) GetMetadata() map[string]interface{} {
	return f.config.Metadata
}

// SetMetadata sets the factory metadata
func (f *ConcreteFactory1) SetMetadata(key string, value interface{}) {
	if f.config.Metadata == nil {
		f.config.Metadata = make(map[string]interface{})
	}
	f.config.Metadata[key] = value
	f.updatedAt = time.Now()
}

// Cleanup performs cleanup operations
func (f *ConcreteFactory1) Cleanup(ctx context.Context) error {
	if !f.active {
		return ErrFactoryInactive
	}
	// Perform cleanup operations
	f.updatedAt = time.Now()
	return nil
}

// ConcreteFactory2 represents another concrete implementation of AbstractFactory
type ConcreteFactory2 struct {
	config    *FactoryConfig
	createdAt time.Time
	updatedAt time.Time
	active    bool
}

// NewConcreteFactory2 creates a new ConcreteFactory2
func NewConcreteFactory2(config *FactoryConfig) *ConcreteFactory2 {
	return &ConcreteFactory2{
		config:    config,
		createdAt: time.Now(),
		updatedAt: time.Now(),
		active:    true,
	}
}

// CreateProductA creates a ProductA
func (f *ConcreteFactory2) CreateProductA() ProductA {
	return NewConcreteProductA("product-a-2", "Product A2", "Description for Product A2", 150.0)
}

// CreateProductB creates a ProductB
func (f *ConcreteFactory2) CreateProductB() ProductB {
	return NewConcreteProductB("product-b-2", "Product B2", "Description for Product B2", 250.0)
}

// CreateProductC creates a ProductC
func (f *ConcreteFactory2) CreateProductC() ProductC {
	return NewConcreteProductC("product-c-2", "Product C2", "Description for Product C2", 350.0)
}

// GetFactoryType returns the factory type
func (f *ConcreteFactory2) GetFactoryType() string {
	return "ConcreteFactory2"
}

// GetFactoryInfo returns factory information
func (f *ConcreteFactory2) GetFactoryInfo() map[string]interface{} {
	return map[string]interface{}{
		"type":        f.GetFactoryType(),
		"name":        f.config.Name,
		"description": f.config.Description,
		"version":     f.config.Version,
		"active":      f.active,
		"created_at":  f.createdAt,
		"updated_at":  f.updatedAt,
		"metadata":    f.config.Metadata,
	}
}

// IsActive returns whether the factory is active
func (f *ConcreteFactory2) IsActive() bool {
	return f.active
}

// SetActive sets the active status
func (f *ConcreteFactory2) SetActive(active bool) {
	f.active = active
	f.updatedAt = time.Now()
}

// GetCreatedAt returns the creation time
func (f *ConcreteFactory2) GetCreatedAt() time.Time {
	return f.createdAt
}

// GetUpdatedAt returns the last update time
func (f *ConcreteFactory2) GetUpdatedAt() time.Time {
	return f.updatedAt
}

// GetMetadata returns the factory metadata
func (f *ConcreteFactory2) GetMetadata() map[string]interface{} {
	return f.config.Metadata
}

// SetMetadata sets the factory metadata
func (f *ConcreteFactory2) SetMetadata(key string, value interface{}) {
	if f.config.Metadata == nil {
		f.config.Metadata = make(map[string]interface{})
	}
	f.config.Metadata[key] = value
	f.updatedAt = time.Now()
}

// Cleanup performs cleanup operations
func (f *ConcreteFactory2) Cleanup(ctx context.Context) error {
	if !f.active {
		return ErrFactoryInactive
	}
	// Perform cleanup operations
	f.updatedAt = time.Now()
	return nil
}
