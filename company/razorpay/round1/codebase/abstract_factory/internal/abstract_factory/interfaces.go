package abstract_factory

import (
	"context"
	"time"
)

// AbstractFactory defines the interface for creating families of related objects
type AbstractFactory interface {
	CreateProductA() ProductA
	CreateProductB() ProductB
	CreateProductC() ProductC
	GetFactoryType() string
	GetFactoryInfo() map[string]interface{}
	IsActive() bool
	SetActive(active bool)
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	Cleanup(ctx context.Context) error
}

// ProductA defines the interface for Product A
type ProductA interface {
	GetID() string
	GetName() string
	GetType() string
	GetDescription() string
	GetPrice() float64
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	IsActive() bool
	SetActive(active bool)
	Process(ctx context.Context) error
	Validate() error
	GetStats() map[string]interface{}
}

// ProductB defines the interface for Product B
type ProductB interface {
	GetID() string
	GetName() string
	GetType() string
	GetDescription() string
	GetPrice() float64
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	IsActive() bool
	SetActive(active bool)
	Process(ctx context.Context) error
	Validate() error
	GetStats() map[string]interface{}
}

// ProductC defines the interface for Product C
type ProductC interface {
	GetID() string
	GetName() string
	GetType() string
	GetDescription() string
	GetPrice() float64
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	GetMetadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	IsActive() bool
	SetActive(active bool)
	Process(ctx context.Context) error
	Validate() error
	GetStats() map[string]interface{}
}

// FactoryRegistry manages the registration and retrieval of factories
type FactoryRegistry interface {
	RegisterFactory(factoryType string, factory AbstractFactory) error
	UnregisterFactory(factoryType string) error
	GetFactory(factoryType string) (AbstractFactory, error)
	ListFactories() []string
	GetFactoryStats(factoryType string) map[string]interface{}
	GetAllFactoryStats() map[string]interface{}
	IsFactoryRegistered(factoryType string) bool
	GetRegistryStats() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// ProductRegistry manages the registration and retrieval of products
type ProductRegistry interface {
	RegisterProduct(productID string, product interface{}) error
	UnregisterProduct(productID string) error
	GetProduct(productID string) (interface{}, error)
	ListProducts() []string
	GetProductStats(productID string) map[string]interface{}
	GetAllProductStats() map[string]interface{}
	IsProductRegistered(productID string) bool
	GetRegistryStats() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// FactoryService provides high-level operations for factory management
type FactoryService interface {
	CreateFactory(factoryType string, config map[string]interface{}) (AbstractFactory, error)
	DestroyFactory(factoryType string) error
	GetFactory(factoryType string) (AbstractFactory, error)
	ListFactories() []string
	GetFactoryStats(factoryType string) map[string]interface{}
	GetAllFactoryStats() map[string]interface{}
	IsFactoryActive(factoryType string) bool
	SetFactoryActive(factoryType string, active bool) error
	GetServiceStats() map[string]interface{}
	GetHealthStatus() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// ProductService provides high-level operations for product management
type ProductService interface {
	CreateProduct(factoryType string, productType string, config map[string]interface{}) (interface{}, error)
	DestroyProduct(productID string) error
	GetProduct(productID string) (interface{}, error)
	ListProducts() []string
	GetProductStats(productID string) map[string]interface{}
	GetAllProductStats() map[string]interface{}
	IsProductActive(productID string) bool
	SetProductActive(productID string, active bool) error
	GetServiceStats() map[string]interface{}
	GetHealthStatus() map[string]interface{}
	Cleanup(ctx context.Context) error
}

// FactoryConfig holds configuration for a factory
type FactoryConfig struct {
	Type        string                 `json:"type" yaml:"type"`
	Name        string                 `json:"name" yaml:"name"`
	Description string                 `json:"description" yaml:"description"`
	Version     string                 `json:"version" yaml:"version"`
	Metadata    map[string]interface{} `json:"metadata" yaml:"metadata"`
	Active      bool                   `json:"active" yaml:"active"`
	CreatedAt   time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" yaml:"updated_at"`
}

// ProductConfig holds configuration for a product
type ProductConfig struct {
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

// ServiceConfig holds configuration for the service
type ServiceConfig struct {
	Name                    string                 `json:"name" yaml:"name"`
	Version                 string                 `json:"version" yaml:"version"`
	Description             string                 `json:"description" yaml:"description"`
	MaxFactories            int                    `json:"max_factories" yaml:"max_factories"`
	MaxProducts             int                    `json:"max_products" yaml:"max_products"`
	CleanupInterval         time.Duration          `json:"cleanup_interval" yaml:"cleanup_interval"`
	ValidationEnabled       bool                   `json:"validation_enabled" yaml:"validation_enabled"`
	CachingEnabled          bool                   `json:"caching_enabled" yaml:"caching_enabled"`
	MonitoringEnabled       bool                   `json:"monitoring_enabled" yaml:"monitoring_enabled"`
	AuditingEnabled         bool                   `json:"auditing_enabled" yaml:"auditing_enabled"`
	SupportedFactoryTypes   []string               `json:"supported_factory_types" yaml:"supported_factory_types"`
	SupportedProductTypes   []string               `json:"supported_product_types" yaml:"supported_product_types"`
	ValidationRules         map[string]interface{} `json:"validation_rules" yaml:"validation_rules"`
	Metadata                map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// FactoryStats holds statistics for a factory
type FactoryStats struct {
	FactoryType     string                 `json:"factory_type" yaml:"factory_type"`
	Active          bool                   `json:"active" yaml:"active"`
	CreatedAt       time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at" yaml:"updated_at"`
	ProductsCreated int                    `json:"products_created" yaml:"products_created"`
	ProductsActive  int                    `json:"products_active" yaml:"products_active"`
	Metadata        map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// ProductStats holds statistics for a product
type ProductStats struct {
	ProductID   string                 `json:"product_id" yaml:"product_id"`
	ProductType string                 `json:"product_type" yaml:"product_type"`
	Active      bool                   `json:"active" yaml:"active"`
	CreatedAt   time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" yaml:"updated_at"`
	Processed   int                    `json:"processed" yaml:"processed"`
	Metadata    map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// ServiceStats holds statistics for the service
type ServiceStats struct {
	ServiceName      string                 `json:"service_name" yaml:"service_name"`
	Version          string                 `json:"version" yaml:"version"`
	Active           bool                   `json:"active" yaml:"active"`
	CreatedAt        time.Time              `json:"created_at" yaml:"created_at"`
	UpdatedAt        time.Time              `json:"updated_at" yaml:"updated_at"`
	FactoriesCount   int                    `json:"factories_count" yaml:"factories_count"`
	ProductsCount    int                    `json:"products_count" yaml:"products_count"`
	Metadata         map[string]interface{} `json:"metadata" yaml:"metadata"`
}

// HealthStatus holds health status information
type HealthStatus struct {
	Status    string                 `json:"status" yaml:"status"`
	Checks    map[string]interface{} `json:"checks" yaml:"checks"`
	Timestamp time.Time              `json:"timestamp" yaml:"timestamp"`
}
