package proxy

import (
	"context"
	"fmt"
	"time"
)

// PaymentService implements the Service interface for payment processing
type PaymentService struct {
	name   string
	config ServiceConfig
}

// NewPaymentService creates a new payment service
func NewPaymentService(config ServiceConfig) *PaymentService {
	return &PaymentService{
		name:   config.Name,
		config: config,
	}
}

// Process processes a payment request
func (ps *PaymentService) Process(ctx context.Context, request interface{}) (interface{}, error) {
	// Simulate payment processing
	time.Sleep(100 * time.Millisecond)
	
	// Mock payment response
	response := map[string]interface{}{
		"transaction_id": fmt.Sprintf("txn_%d", time.Now().Unix()),
		"status":         "success",
		"amount":         100.50,
		"currency":       "INR",
		"timestamp":      time.Now(),
	}
	
	return response, nil
}

// GetName returns the service name
func (ps *PaymentService) GetName() string {
	return ps.name
}

// IsHealthy checks if the service is healthy
func (ps *PaymentService) IsHealthy(ctx context.Context) bool {
	// Simulate health check
	return true
}

// NotificationService implements the Service interface for notifications
type NotificationService struct {
	name   string
	config ServiceConfig
}

// NewNotificationService creates a new notification service
func NewNotificationService(config ServiceConfig) *NotificationService {
	return &NotificationService{
		name:   config.Name,
		config: config,
	}
}

// Process processes a notification request
func (ns *NotificationService) Process(ctx context.Context, request interface{}) (interface{}, error) {
	// Simulate notification sending
	time.Sleep(50 * time.Millisecond)
	
	// Mock notification response
	response := map[string]interface{}{
		"message_id": fmt.Sprintf("msg_%d", time.Now().Unix()),
		"status":     "sent",
		"channel":    "email",
		"timestamp":  time.Now(),
	}
	
	return response, nil
}

// GetName returns the service name
func (ns *NotificationService) GetName() string {
	return ns.name
}

// IsHealthy checks if the service is healthy
func (ns *NotificationService) IsHealthy(ctx context.Context) bool {
	// Simulate health check
	return true
}

// UserService implements the Service interface for user management
type UserService struct {
	name   string
	config ServiceConfig
}

// NewUserService creates a new user service
func NewUserService(config ServiceConfig) *UserService {
	return &UserService{
		name:   config.Name,
		config: config,
	}
}

// Process processes a user request
func (us *UserService) Process(ctx context.Context, request interface{}) (interface{}, error) {
	// Simulate user processing
	time.Sleep(75 * time.Millisecond)
	
	// Mock user response
	response := map[string]interface{}{
		"user_id":   fmt.Sprintf("user_%d", time.Now().Unix()),
		"username":  "testuser",
		"email":     "test@example.com",
		"status":    "active",
		"timestamp": time.Now(),
	}
	
	return response, nil
}

// GetName returns the service name
func (us *UserService) GetName() string {
	return us.name
}

// IsHealthy checks if the service is healthy
func (us *UserService) IsHealthy(ctx context.Context) bool {
	// Simulate health check
	return true
}

// OrderService implements the Service interface for order management
type OrderService struct {
	name   string
	config ServiceConfig
}

// NewOrderService creates a new order service
func NewOrderService(config ServiceConfig) *OrderService {
	return &OrderService{
		name:   config.Name,
		config: config,
	}
}

// Process processes an order request
func (os *OrderService) Process(ctx context.Context, request interface{}) (interface{}, error) {
	// Simulate order processing
	time.Sleep(150 * time.Millisecond)
	
	// Mock order response
	response := map[string]interface{}{
		"order_id":   fmt.Sprintf("order_%d", time.Now().Unix()),
		"status":     "confirmed",
		"total":      250.75,
		"currency":   "INR",
		"timestamp":  time.Now(),
	}
	
	return response, nil
}

// GetName returns the service name
func (os *OrderService) GetName() string {
	return os.name
}

// IsHealthy checks if the service is healthy
func (os *OrderService) IsHealthy(ctx context.Context) bool {
	// Simulate health check
	return true
}

// InventoryService implements the Service interface for inventory management
type InventoryService struct {
	name   string
	config ServiceConfig
}

// NewInventoryService creates a new inventory service
func NewInventoryService(config ServiceConfig) *InventoryService {
	return &InventoryService{
		name:   config.Name,
		config: config,
	}
}

// Process processes an inventory request
func (is *InventoryService) Process(ctx context.Context, request interface{}) (interface{}, error) {
	// Simulate inventory processing
	time.Sleep(80 * time.Millisecond)
	
	// Mock inventory response
	response := map[string]interface{}{
		"product_id": fmt.Sprintf("prod_%d", time.Now().Unix()),
		"quantity":   100,
		"available":  true,
		"timestamp":  time.Now(),
	}
	
	return response, nil
}

// GetName returns the service name
func (is *InventoryService) GetName() string {
	return is.name
}

// IsHealthy checks if the service is healthy
func (is *InventoryService) IsHealthy(ctx context.Context) bool {
	// Simulate health check
	return true
}
