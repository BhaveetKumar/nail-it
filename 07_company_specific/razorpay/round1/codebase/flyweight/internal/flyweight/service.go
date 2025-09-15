package flyweight

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// FlyweightService provides high-level operations using flyweights
type FlyweightService struct {
	factory    FlyweightFactory
	cache      Cache
	database   Database
	logger     Logger
	metrics    Metrics
	config     FlyweightConfig
	mu         sync.RWMutex
}

// NewFlyweightService creates a new flyweight service
func NewFlyweightService(
	factory FlyweightFactory,
	cache Cache,
	database Database,
	logger Logger,
	metrics Metrics,
	config FlyweightConfig,
) *FlyweightService {
	return &FlyweightService{
		factory:  factory,
		cache:    cache,
		database: database,
		logger:   logger,
		metrics:  metrics,
		config:   config,
	}
}

// GetProduct retrieves a product using flyweight pattern
func (fs *FlyweightService) GetProduct(ctx context.Context, productID string) (*ProductInfo, error) {
	start := time.Now()
	
	// Try to get from flyweight factory first
	flyweight, err := fs.factory.GetFlyweight(productID)
	if err != nil {
		// If not found, try to load from database
		productData, err := fs.loadProductFromDatabase(ctx, productID)
		if err != nil {
			fs.logger.Error("Failed to load product from database", "product_id", productID, "error", err)
			return nil, fmt.Errorf("product not found: %w", err)
		}
		
		// Create flyweight from database data
		flyweight, err = fs.factory.CreateFlyweight(productID, productData)
		if err != nil {
			fs.logger.Error("Failed to create product flyweight", "product_id", productID, "error", err)
			return nil, fmt.Errorf("failed to create product flyweight: %w", err)
		}
	}
	
	// Convert flyweight to product info
	productInfo := fs.convertFlyweightToProductInfo(flyweight)
	
	duration := time.Since(start)
	fs.metrics.RecordTiming("product_retrieval_duration", duration, map[string]string{"source": "flyweight"})
	fs.metrics.IncrementCounter("product_retrieved", map[string]string{"source": "flyweight"})
	
	fs.logger.Debug("Product retrieved", "product_id", productID, "duration", duration)
	
	return productInfo, nil
}

// GetUser retrieves a user using flyweight pattern
func (fs *FlyweightService) GetUser(ctx context.Context, userID string) (*UserInfo, error) {
	start := time.Now()
	
	// Try to get from flyweight factory first
	flyweight, err := fs.factory.GetFlyweight(userID)
	if err != nil {
		// If not found, try to load from database
		userData, err := fs.loadUserFromDatabase(ctx, userID)
		if err != nil {
			fs.logger.Error("Failed to load user from database", "user_id", userID, "error", err)
			return nil, fmt.Errorf("user not found: %w", err)
		}
		
		// Create flyweight from database data
		flyweight, err = fs.factory.CreateFlyweight(userID, userData)
		if err != nil {
			fs.logger.Error("Failed to create user flyweight", "user_id", userID, "error", err)
			return nil, fmt.Errorf("failed to create user flyweight: %w", err)
		}
	}
	
	// Convert flyweight to user info
	userInfo := fs.convertFlyweightToUserInfo(flyweight)
	
	duration := time.Since(start)
	fs.metrics.RecordTiming("user_retrieval_duration", duration, map[string]string{"source": "flyweight"})
	fs.metrics.IncrementCounter("user_retrieved", map[string]string{"source": "flyweight"})
	
	fs.logger.Debug("User retrieved", "user_id", userID, "duration", duration)
	
	return userInfo, nil
}

// GetOrder retrieves an order using flyweight pattern
func (fs *FlyweightService) GetOrder(ctx context.Context, orderID string) (*OrderInfo, error) {
	start := time.Now()
	
	// Try to get from flyweight factory first
	flyweight, err := fs.factory.GetFlyweight(orderID)
	if err != nil {
		// If not found, try to load from database
		orderData, err := fs.loadOrderFromDatabase(ctx, orderID)
		if err != nil {
			fs.logger.Error("Failed to load order from database", "order_id", orderID, "error", err)
			return nil, fmt.Errorf("order not found: %w", err)
		}
		
		// Create flyweight from database data
		flyweight, err = fs.factory.CreateFlyweight(orderID, orderData)
		if err != nil {
			fs.logger.Error("Failed to create order flyweight", "order_id", orderID, "error", err)
			return nil, fmt.Errorf("failed to create order flyweight: %w", err)
		}
	}
	
	// Convert flyweight to order info
	orderInfo := fs.convertFlyweightToOrderInfo(flyweight)
	
	duration := time.Since(start)
	fs.metrics.RecordTiming("order_retrieval_duration", duration, map[string]string{"source": "flyweight"})
	fs.metrics.IncrementCounter("order_retrieved", map[string]string{"source": "flyweight"})
	
	fs.logger.Debug("Order retrieved", "order_id", orderID, "duration", duration)
	
	return orderInfo, nil
}

// GetNotificationTemplate retrieves a notification template using flyweight pattern
func (fs *FlyweightService) GetNotificationTemplate(ctx context.Context, templateID string) (*NotificationTemplate, error) {
	start := time.Now()
	
	// Try to get from flyweight factory first
	flyweight, err := fs.factory.GetFlyweight(templateID)
	if err != nil {
		// If not found, try to load from database
		templateData, err := fs.loadNotificationTemplateFromDatabase(ctx, templateID)
		if err != nil {
			fs.logger.Error("Failed to load notification template from database", "template_id", templateID, "error", err)
			return nil, fmt.Errorf("notification template not found: %w", err)
		}
		
		// Create flyweight from database data
		flyweight, err = fs.factory.CreateFlyweight(templateID, templateData)
		if err != nil {
			fs.logger.Error("Failed to create notification template flyweight", "template_id", templateID, "error", err)
			return nil, fmt.Errorf("failed to create notification template flyweight: %w", err)
		}
	}
	
	// Convert flyweight to notification template
	template := fs.convertFlyweightToNotificationTemplate(flyweight)
	
	duration := time.Since(start)
	fs.metrics.RecordTiming("notification_template_retrieval_duration", duration, map[string]string{"source": "flyweight"})
	fs.metrics.IncrementCounter("notification_template_retrieved", map[string]string{"source": "flyweight"})
	
	fs.logger.Debug("Notification template retrieved", "template_id", templateID, "duration", duration)
	
	return template, nil
}

// GetConfiguration retrieves a configuration using flyweight pattern
func (fs *FlyweightService) GetConfiguration(ctx context.Context, configKey string) (*ConfigurationInfo, error) {
	start := time.Now()
	
	// Try to get from flyweight factory first
	flyweight, err := fs.factory.GetFlyweight(configKey)
	if err != nil {
		// If not found, try to load from database
		configData, err := fs.loadConfigurationFromDatabase(ctx, configKey)
		if err != nil {
			fs.logger.Error("Failed to load configuration from database", "config_key", configKey, "error", err)
			return nil, fmt.Errorf("configuration not found: %w", err)
		}
		
		// Create flyweight from database data
		flyweight, err = fs.factory.CreateFlyweight(configKey, configData)
		if err != nil {
			fs.logger.Error("Failed to create configuration flyweight", "config_key", configKey, "error", err)
			return nil, fmt.Errorf("failed to create configuration flyweight: %w", err)
		}
	}
	
	// Convert flyweight to configuration info
	configInfo := fs.convertFlyweightToConfigurationInfo(flyweight)
	
	duration := time.Since(start)
	fs.metrics.RecordTiming("configuration_retrieval_duration", duration, map[string]string{"source": "flyweight"})
	fs.metrics.IncrementCounter("configuration_retrieved", map[string]string{"source": "flyweight"})
	
	fs.logger.Debug("Configuration retrieved", "config_key", configKey, "duration", duration)
	
	return configInfo, nil
}

// GetFactoryStats returns factory statistics
func (fs *FlyweightService) GetFactoryStats() FactoryStats {
	return fs.factory.GetFactoryStats()
}

// ClearUnusedFlyweights clears unused flyweights
func (fs *FlyweightService) ClearUnusedFlyweights() {
	fs.factory.ClearUnusedFlyweights()
}

// Helper methods

func (fs *FlyweightService) loadProductFromDatabase(ctx context.Context, productID string) (map[string]interface{}, error) {
	// Simulate database load
	time.Sleep(50 * time.Millisecond)
	
	// Mock product data
	return map[string]interface{}{
		"type":        "product",
		"name":        "Sample Product",
		"description": "A sample product description",
		"category":    "electronics",
		"brand":       "SampleBrand",
		"base_price":  99.99,
		"currency":    "INR",
		"attributes": map[string]interface{}{
			"color": "black",
			"size":  "medium",
		},
	}, nil
}

func (fs *FlyweightService) loadUserFromDatabase(ctx context.Context, userID string) (map[string]interface{}, error) {
	// Simulate database load
	time.Sleep(40 * time.Millisecond)
	
	// Mock user data
	return map[string]interface{}{
		"type":    "user",
		"username": "sampleuser",
		"email":   "user@example.com",
		"profile": map[string]interface{}{
			"name": "Sample User",
			"age":  30,
		},
		"preferences": map[string]interface{}{
			"theme": "dark",
			"lang":  "en",
		},
	}, nil
}

func (fs *FlyweightService) loadOrderFromDatabase(ctx context.Context, orderID string) (map[string]interface{}, error) {
	// Simulate database load
	time.Sleep(60 * time.Millisecond)
	
	// Mock order data
	return map[string]interface{}{
		"type":     "order",
		"status":   "pending",
		"priority": "normal",
		"metadata": map[string]interface{}{
			"source": "web",
			"channel": "online",
		},
	}, nil
}

func (fs *FlyweightService) loadNotificationTemplateFromDatabase(ctx context.Context, templateID string) (map[string]interface{}, error) {
	// Simulate database load
	time.Sleep(30 * time.Millisecond)
	
	// Mock notification template data
	return map[string]interface{}{
		"type":     "notification",
		"template": "order_confirmation",
		"subject":  "Order Confirmation",
		"body":     "Your order has been confirmed. Order ID: {{order_id}}",
		"channels": []string{"email", "sms"},
		"metadata": map[string]interface{}{
			"category": "order",
			"priority": "high",
		},
	}, nil
}

func (fs *FlyweightService) loadConfigurationFromDatabase(ctx context.Context, configKey string) (map[string]interface{}, error) {
	// Simulate database load
	time.Sleep(20 * time.Millisecond)
	
	// Mock configuration data
	return map[string]interface{}{
		"type":        "configuration",
		"key":         configKey,
		"value":       "sample_value",
		"description": "Sample configuration",
		"category":    "general",
		"metadata": map[string]interface{}{
			"version": "1.0",
			"env":     "production",
		},
	}, nil
}

func (fs *FlyweightService) convertFlyweightToProductInfo(flyweight Flyweight) *ProductInfo {
	intrinsicState := flyweight.GetIntrinsicState()
	
	return &ProductInfo{
		ID:          flyweight.GetID(),
		Name:        intrinsicState["name"].(string),
		Description: intrinsicState["description"].(string),
		Category:    intrinsicState["category"].(string),
		Brand:       intrinsicState["brand"].(string),
		BasePrice:   intrinsicState["base_price"].(float64),
		Currency:    intrinsicState["currency"].(string),
		Attributes:  intrinsicState["attributes"].(map[string]interface{}),
		IsShared:    flyweight.IsShared(),
		CreatedAt:   flyweight.GetCreatedAt(),
		LastAccessed: flyweight.GetLastAccessed(),
	}
}

func (fs *FlyweightService) convertFlyweightToUserInfo(flyweight Flyweight) *UserInfo {
	intrinsicState := flyweight.GetIntrinsicState()
	
	return &UserInfo{
		ID:           flyweight.GetID(),
		Username:     intrinsicState["username"].(string),
		Email:        intrinsicState["email"].(string),
		Profile:      intrinsicState["profile"].(map[string]interface{}),
		Preferences:  intrinsicState["preferences"].(map[string]interface{}),
		IsShared:     flyweight.IsShared(),
		CreatedAt:    flyweight.GetCreatedAt(),
		LastAccessed: flyweight.GetLastAccessed(),
	}
}

func (fs *FlyweightService) convertFlyweightToOrderInfo(flyweight Flyweight) *OrderInfo {
	intrinsicState := flyweight.GetIntrinsicState()
	
	return &OrderInfo{
		ID:           flyweight.GetID(),
		Status:       intrinsicState["status"].(string),
		Priority:     intrinsicState["priority"].(string),
		Metadata:     intrinsicState["metadata"].(map[string]interface{}),
		IsShared:     flyweight.IsShared(),
		CreatedAt:    flyweight.GetCreatedAt(),
		LastAccessed: flyweight.GetLastAccessed(),
	}
}

func (fs *FlyweightService) convertFlyweightToNotificationTemplate(flyweight Flyweight) *NotificationTemplate {
	intrinsicState := flyweight.GetIntrinsicState()
	
	return &NotificationTemplate{
		ID:           flyweight.GetID(),
		Template:     intrinsicState["template"].(string),
		Subject:      intrinsicState["subject"].(string),
		Body:         intrinsicState["body"].(string),
		Channels:     intrinsicState["channels"].([]string),
		Metadata:     intrinsicState["metadata"].(map[string]interface{}),
		IsShared:     flyweight.IsShared(),
		CreatedAt:    flyweight.GetCreatedAt(),
		LastAccessed: flyweight.GetLastAccessed(),
	}
}

func (fs *FlyweightService) convertFlyweightToConfigurationInfo(flyweight Flyweight) *ConfigurationInfo {
	intrinsicState := flyweight.GetIntrinsicState()
	
	return &ConfigurationInfo{
		ID:           flyweight.GetID(),
		Key:          intrinsicState["key"].(string),
		Value:        intrinsicState["value"],
		Description:  intrinsicState["description"].(string),
		Category:     intrinsicState["category"].(string),
		Metadata:     intrinsicState["metadata"].(map[string]interface{}),
		IsShared:     flyweight.IsShared(),
		CreatedAt:    flyweight.GetCreatedAt(),
		LastAccessed: flyweight.GetLastAccessed(),
	}
}

// Response models

type ProductInfo struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	Category     string                 `json:"category"`
	Brand        string                 `json:"brand"`
	BasePrice    float64                `json:"base_price"`
	Currency     string                 `json:"currency"`
	Attributes   map[string]interface{} `json:"attributes"`
	IsShared     bool                   `json:"is_shared"`
	CreatedAt    time.Time              `json:"created_at"`
	LastAccessed time.Time              `json:"last_accessed"`
}

type UserInfo struct {
	ID           string                 `json:"id"`
	Username     string                 `json:"username"`
	Email        string                 `json:"email"`
	Profile      map[string]interface{} `json:"profile"`
	Preferences  map[string]interface{} `json:"preferences"`
	IsShared     bool                   `json:"is_shared"`
	CreatedAt    time.Time              `json:"created_at"`
	LastAccessed time.Time              `json:"last_accessed"`
}

type OrderInfo struct {
	ID           string                 `json:"id"`
	Status       string                 `json:"status"`
	Priority     string                 `json:"priority"`
	Metadata     map[string]interface{} `json:"metadata"`
	IsShared     bool                   `json:"is_shared"`
	CreatedAt    time.Time              `json:"created_at"`
	LastAccessed time.Time              `json:"last_accessed"`
}

type NotificationTemplate struct {
	ID           string                 `json:"id"`
	Template     string                 `json:"template"`
	Subject      string                 `json:"subject"`
	Body         string                 `json:"body"`
	Channels     []string               `json:"channels"`
	Metadata     map[string]interface{} `json:"metadata"`
	IsShared     bool                   `json:"is_shared"`
	CreatedAt    time.Time              `json:"created_at"`
	LastAccessed time.Time              `json:"last_accessed"`
}

type ConfigurationInfo struct {
	ID           string                 `json:"id"`
	Key          string                 `json:"key"`
	Value        interface{}            `json:"value"`
	Description  string                 `json:"description"`
	Category     string                 `json:"category"`
	Metadata     map[string]interface{} `json:"metadata"`
	IsShared     bool                   `json:"is_shared"`
	CreatedAt    time.Time              `json:"created_at"`
	LastAccessed time.Time              `json:"last_accessed"`
}
