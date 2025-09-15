package facade

import (
	"context"
	"fmt"
	"time"
)

// MockPaymentService implements PaymentService interface
type MockPaymentService struct {
	logger Logger
}

// NewMockPaymentService creates a new mock payment service
func NewMockPaymentService(logger Logger) *MockPaymentService {
	return &MockPaymentService{logger: logger}
}

// ProcessPayment processes a payment
func (mps *MockPaymentService) ProcessPayment(ctx context.Context, request PaymentRequest) (*PaymentResponse, error) {
	// Simulate payment processing
	time.Sleep(100 * time.Millisecond)
	
	response := &PaymentResponse{
		ID:            fmt.Sprintf("pay_%d", time.Now().Unix()),
		Status:        "success",
		TransactionID: fmt.Sprintf("txn_%d", time.Now().Unix()),
		Amount:        request.Amount,
		Currency:      request.Currency,
		Gateway:       "razorpay",
		Metadata:      request.Metadata,
		CreatedAt:     request.CreatedAt,
		ProcessedAt:   time.Now(),
	}
	
	mps.logger.Info("Payment processed", "payment_id", response.ID, "amount", request.Amount)
	return response, nil
}

// RefundPayment refunds a payment
func (mps *MockPaymentService) RefundPayment(ctx context.Context, transactionID string, amount float64) error {
	// Simulate refund processing
	time.Sleep(50 * time.Millisecond)
	
	mps.logger.Info("Payment refunded", "transaction_id", transactionID, "amount", amount)
	return nil
}

// GetPaymentStatus gets payment status
func (mps *MockPaymentService) GetPaymentStatus(ctx context.Context, transactionID string) (*PaymentStatus, error) {
	// Simulate status check
	time.Sleep(30 * time.Millisecond)
	
	return &PaymentStatus{
		TransactionID: transactionID,
		Status:        "success",
		Amount:        100.50,
		Currency:      "INR",
		LastUpdated:   time.Now(),
	}, nil
}

// GetPaymentHistory gets payment history
func (mps *MockPaymentService) GetPaymentHistory(ctx context.Context, userID string) ([]*PaymentHistory, error) {
	// Simulate history retrieval
	time.Sleep(40 * time.Millisecond)
	
	history := []*PaymentHistory{
		{
			ID:          fmt.Sprintf("pay_%d", time.Now().Unix()),
			Amount:      100.50,
			Currency:    "INR",
			Status:      "success",
			Description: "Order payment",
			CreatedAt:   time.Now().Add(-24 * time.Hour),
		},
		{
			ID:          fmt.Sprintf("pay_%d", time.Now().Unix()-1),
			Amount:      250.75,
			Currency:    "INR",
			Status:      "success",
			Description: "Order payment",
			CreatedAt:   time.Now().Add(-48 * time.Hour),
		},
	}
	
	return history, nil
}

// MockNotificationService implements NotificationService interface
type MockNotificationService struct {
	logger Logger
}

// NewMockNotificationService creates a new mock notification service
func NewMockNotificationService(logger Logger) *MockNotificationService {
	return &MockNotificationService{logger: logger}
}

// SendEmail sends an email
func (mns *MockNotificationService) SendEmail(ctx context.Context, request EmailRequest) error {
	// Simulate email sending
	time.Sleep(200 * time.Millisecond)
	
	mns.logger.Info("Email sent", "to", request.To, "subject", request.Subject)
	return nil
}

// SendSMS sends an SMS
func (mns *MockNotificationService) SendSMS(ctx context.Context, request SMSRequest) error {
	// Simulate SMS sending
	time.Sleep(100 * time.Millisecond)
	
	mns.logger.Info("SMS sent", "to", request.To, "message", request.Message)
	return nil
}

// SendPushNotification sends a push notification
func (mns *MockNotificationService) SendPushNotification(ctx context.Context, request PushRequest) error {
	// Simulate push notification sending
	time.Sleep(80 * time.Millisecond)
	
	mns.logger.Info("Push notification sent", "user_id", request.UserID, "title", request.Title)
	return nil
}

// SendBulkNotification sends bulk notifications
func (mns *MockNotificationService) SendBulkNotification(ctx context.Context, request BulkNotificationRequest) error {
	// Simulate bulk notification sending
	time.Sleep(300 * time.Millisecond)
	
	mns.logger.Info("Bulk notification sent", "recipients", len(request.Recipients), "type", request.Type)
	return nil
}

// MockUserService implements UserService interface
type MockUserService struct {
	logger Logger
}

// NewMockUserService creates a new mock user service
func NewMockUserService(logger Logger) *MockUserService {
	return &MockUserService{logger: logger}
}

// CreateUser creates a user
func (mus *MockUserService) CreateUser(ctx context.Context, request CreateUserRequest) (*User, error) {
	// Simulate user creation
	time.Sleep(75 * time.Millisecond)
	
	user := &User{
		ID:        fmt.Sprintf("user_%d", time.Now().Unix()),
		Username:  request.Username,
		Email:     request.Email,
		Profile:   request.Profile,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	
	mus.logger.Info("User created", "user_id", user.ID, "username", user.Username)
	return user, nil
}

// GetUser gets a user
func (mus *MockUserService) GetUser(ctx context.Context, userID string) (*User, error) {
	// Simulate user retrieval
	time.Sleep(50 * time.Millisecond)
	
	user := &User{
		ID:        userID,
		Username:  "testuser",
		Email:     "test@example.com",
		Profile:   map[string]interface{}{"name": "Test User"},
		CreatedAt: time.Now().Add(-30 * 24 * time.Hour),
		UpdatedAt: time.Now().Add(-1 * time.Hour),
	}
	
	return user, nil
}

// UpdateUser updates a user
func (mus *MockUserService) UpdateUser(ctx context.Context, userID string, request UpdateUserRequest) (*User, error) {
	// Simulate user update
	time.Sleep(60 * time.Millisecond)
	
	user := &User{
		ID:        userID,
		Username:  request.Username,
		Email:     request.Email,
		Profile:   request.Profile,
		CreatedAt: time.Now().Add(-30 * 24 * time.Hour),
		UpdatedAt: time.Now(),
	}
	
	mus.logger.Info("User updated", "user_id", userID)
	return user, nil
}

// DeleteUser deletes a user
func (mus *MockUserService) DeleteUser(ctx context.Context, userID string) error {
	// Simulate user deletion
	time.Sleep(40 * time.Millisecond)
	
	mus.logger.Info("User deleted", "user_id", userID)
	return nil
}

// AuthenticateUser authenticates a user
func (mus *MockUserService) AuthenticateUser(ctx context.Context, request AuthRequest) (*AuthResponse, error) {
	// Simulate authentication
	time.Sleep(80 * time.Millisecond)
	
	user := &User{
		ID:        fmt.Sprintf("user_%d", time.Now().Unix()),
		Username:  request.Username,
		Email:     "test@example.com",
		Profile:   map[string]interface{}{"name": "Test User"},
		CreatedAt: time.Now().Add(-30 * 24 * time.Hour),
		UpdatedAt: time.Now(),
	}
	
	response := &AuthResponse{
		Token:     fmt.Sprintf("token_%d", time.Now().Unix()),
		User:      user,
		ExpiresAt: time.Now().Add(24 * time.Hour),
	}
	
	mus.logger.Info("User authenticated", "user_id", user.ID, "username", user.Username)
	return response, nil
}

// MockOrderService implements OrderService interface
type MockOrderService struct {
	logger Logger
}

// NewMockOrderService creates a new mock order service
func NewMockOrderService(logger Logger) *MockOrderService {
	return &MockOrderService{logger: logger}
}

// CreateOrder creates an order
func (mos *MockOrderService) CreateOrder(ctx context.Context, request CreateOrderRequest) (*Order, error) {
	// Simulate order creation
	time.Sleep(120 * time.Millisecond)
	
	// Calculate total
	var total float64
	for _, item := range request.Items {
		total += item.Price * float64(item.Quantity)
	}
	
	order := &Order{
		ID:              fmt.Sprintf("order_%d", time.Now().Unix()),
		UserID:          request.UserID,
		Items:           request.Items,
		Total:           total,
		Status:          "pending",
		ShippingAddress: request.ShippingAddress,
		BillingAddress:  request.BillingAddress,
		Metadata:        request.Metadata,
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}
	
	mos.logger.Info("Order created", "order_id", order.ID, "user_id", request.UserID, "total", total)
	return order, nil
}

// GetOrder gets an order
func (mos *MockOrderService) GetOrder(ctx context.Context, orderID string) (*Order, error) {
	// Simulate order retrieval
	time.Sleep(60 * time.Millisecond)
	
	order := &Order{
		ID:     orderID,
		UserID: "user_123",
		Items: []OrderItem{
			{ProductID: "prod_1", Quantity: 2, Price: 50.0},
			{ProductID: "prod_2", Quantity: 1, Price: 100.0},
		},
		Total:           200.0,
		Status:          "confirmed",
		ShippingAddress: map[string]interface{}{"address": "123 Main St"},
		BillingAddress:  map[string]interface{}{"address": "123 Main St"},
		Metadata:        map[string]interface{}{},
		CreatedAt:       time.Now().Add(-2 * time.Hour),
		UpdatedAt:       time.Now().Add(-1 * time.Hour),
	}
	
	return order, nil
}

// UpdateOrder updates an order
func (mos *MockOrderService) UpdateOrder(ctx context.Context, orderID string, request UpdateOrderRequest) (*Order, error) {
	// Simulate order update
	time.Sleep(70 * time.Millisecond)
	
	order := &Order{
		ID:              orderID,
		UserID:          "user_123",
		Items:           []OrderItem{},
		Total:           200.0,
		Status:          request.Status,
		ShippingAddress: request.ShippingAddress,
		BillingAddress:  request.BillingAddress,
		Metadata:        request.Metadata,
		CreatedAt:       time.Now().Add(-2 * time.Hour),
		UpdatedAt:       time.Now(),
	}
	
	mos.logger.Info("Order updated", "order_id", orderID, "status", request.Status)
	return order, nil
}

// CancelOrder cancels an order
func (mos *MockOrderService) CancelOrder(ctx context.Context, orderID string) error {
	// Simulate order cancellation
	time.Sleep(50 * time.Millisecond)
	
	mos.logger.Info("Order cancelled", "order_id", orderID)
	return nil
}

// GetOrderHistory gets order history
func (mos *MockOrderService) GetOrderHistory(ctx context.Context, userID string) ([]*Order, error) {
	// Simulate order history retrieval
	time.Sleep(80 * time.Millisecond)
	
	orders := []*Order{
		{
			ID:     fmt.Sprintf("order_%d", time.Now().Unix()),
			UserID: userID,
			Items:  []OrderItem{{ProductID: "prod_1", Quantity: 1, Price: 50.0}},
			Total:  50.0,
			Status: "delivered",
			CreatedAt: time.Now().Add(-24 * time.Hour),
			UpdatedAt: time.Now().Add(-12 * time.Hour),
		},
		{
			ID:     fmt.Sprintf("order_%d", time.Now().Unix()-1),
			UserID: userID,
			Items:  []OrderItem{{ProductID: "prod_2", Quantity: 2, Price: 100.0}},
			Total:  200.0,
			Status: "shipped",
			CreatedAt: time.Now().Add(-48 * time.Hour),
			UpdatedAt: time.Now().Add(-24 * time.Hour),
		},
	}
	
	return orders, nil
}

// MockInventoryService implements InventoryService interface
type MockInventoryService struct {
	logger Logger
}

// NewMockInventoryService creates a new mock inventory service
func NewMockInventoryService(logger Logger) *MockInventoryService {
	return &MockInventoryService{logger: logger}
}

// CheckAvailability checks product availability
func (mis *MockInventoryService) CheckAvailability(ctx context.Context, productID string, quantity int) (bool, error) {
	// Simulate availability check
	time.Sleep(40 * time.Millisecond)
	
	// Mock: product is available if quantity <= 100
	available := quantity <= 100
	mis.logger.Info("Availability checked", "product_id", productID, "quantity", quantity, "available", available)
	return available, nil
}

// ReserveProduct reserves a product
func (mis *MockInventoryService) ReserveProduct(ctx context.Context, productID string, quantity int) error {
	// Simulate product reservation
	time.Sleep(60 * time.Millisecond)
	
	mis.logger.Info("Product reserved", "product_id", productID, "quantity", quantity)
	return nil
}

// ReleaseProduct releases a product
func (mis *MockInventoryService) ReleaseProduct(ctx context.Context, productID string, quantity int) error {
	// Simulate product release
	time.Sleep(50 * time.Millisecond)
	
	mis.logger.Info("Product released", "product_id", productID, "quantity", quantity)
	return nil
}

// UpdateStock updates product stock
func (mis *MockInventoryService) UpdateStock(ctx context.Context, productID string, quantity int) error {
	// Simulate stock update
	time.Sleep(70 * time.Millisecond)
	
	mis.logger.Info("Stock updated", "product_id", productID, "quantity", quantity)
	return nil
}

// GetProductInfo gets product information
func (mis *MockInventoryService) GetProductInfo(ctx context.Context, productID string) (*Product, error) {
	// Simulate product info retrieval
	time.Sleep(50 * time.Millisecond)
	
	product := &Product{
		ID:          productID,
		Name:        "Test Product",
		Description: "A test product",
		Price:       50.0,
		Stock:       100,
		Category:    "electronics",
		Metadata:    map[string]interface{}{"brand": "TestBrand"},
		CreatedAt:   time.Now().Add(-30 * 24 * time.Hour),
		UpdatedAt:   time.Now().Add(-1 * time.Hour),
	}
	
	return product, nil
}

// MockAnalyticsService implements AnalyticsService interface
type MockAnalyticsService struct {
	logger Logger
}

// NewMockAnalyticsService creates a new mock analytics service
func NewMockAnalyticsService(logger Logger) *MockAnalyticsService {
	return &MockAnalyticsService{logger: logger}
}

// TrackEvent tracks an event
func (mas *MockAnalyticsService) TrackEvent(ctx context.Context, event Event) error {
	// Simulate event tracking
	time.Sleep(30 * time.Millisecond)
	
	mas.logger.Info("Event tracked", "event_id", event.ID, "event_type", event.EventType, "user_id", event.UserID)
	return nil
}

// GetUserAnalytics gets user analytics
func (mas *MockAnalyticsService) GetUserAnalytics(ctx context.Context, userID string) (*UserAnalytics, error) {
	// Simulate analytics retrieval
	time.Sleep(60 * time.Millisecond)
	
	analytics := &UserAnalytics{
		UserID:           userID,
		TotalEvents:      150,
		LastActivity:     time.Now().Add(-1 * time.Hour),
		FavoriteCategory: "electronics",
		TotalSpent:       1250.75,
	}
	
	return analytics, nil
}

// GetProductAnalytics gets product analytics
func (mas *MockAnalyticsService) GetProductAnalytics(ctx context.Context, productID string) (*ProductAnalytics, error) {
	// Simulate product analytics retrieval
	time.Sleep(50 * time.Millisecond)
	
	analytics := &ProductAnalytics{
		ProductID:      productID,
		Views:          500,
		Purchases:      25,
		Revenue:        1250.0,
		LastViewed:     time.Now().Add(-2 * time.Hour),
		ConversionRate: 5.0,
	}
	
	return analytics, nil
}

// GetSalesReport gets sales report
func (mas *MockAnalyticsService) GetSalesReport(ctx context.Context, request ReportRequest) (*SalesReport, error) {
	// Simulate report generation
	time.Sleep(200 * time.Millisecond)
	
	report := &SalesReport{
		Period:             fmt.Sprintf("%s to %s", request.StartDate.Format("2006-01-02"), request.EndDate.Format("2006-01-02")),
		TotalSales:         50000.0,
		TotalOrders:        250,
		AverageOrderValue:  200.0,
		TopProducts:        []Product{},
		GeneratedAt:        time.Now(),
	}
	
	return report, nil
}

// MockAuditService implements AuditService interface
type MockAuditService struct {
	logger Logger
}

// NewMockAuditService creates a new mock audit service
func NewMockAuditService(logger Logger) *MockAuditService {
	return &MockAuditService{logger: logger}
}

// LogAction logs an action
func (mas *MockAuditService) LogAction(ctx context.Context, action AuditAction) error {
	// Simulate audit logging
	time.Sleep(20 * time.Millisecond)
	
	mas.logger.Info("Action logged", "action_id", action.ID, "action", action.Action, "user_id", action.UserID)
	return nil
}

// GetAuditLogs gets audit logs
func (mas *MockAuditService) GetAuditLogs(ctx context.Context, request AuditLogRequest) ([]*AuditLog, error) {
	// Simulate audit log retrieval
	time.Sleep(80 * time.Millisecond)
	
	logs := []*AuditLog{
		{
			ID:        fmt.Sprintf("audit_%d", time.Now().Unix()),
			UserID:    request.UserID,
			Action:    "login",
			Resource:  "authentication",
			Details:   map[string]interface{}{"ip": "192.168.1.1"},
			IP:        "192.168.1.1",
			UserAgent: "Mozilla/5.0",
			Timestamp: time.Now().Add(-1 * time.Hour),
		},
		{
			ID:        fmt.Sprintf("audit_%d", time.Now().Unix()-1),
			UserID:    request.UserID,
			Action:    "order_created",
			Resource:  "order",
			Details:   map[string]interface{}{"order_id": "order_123"},
			IP:        "192.168.1.1",
			UserAgent: "Mozilla/5.0",
			Timestamp: time.Now().Add(-2 * time.Hour),
		},
	}
	
	return logs, nil
}

// GetUserActivity gets user activity
func (mas *MockAuditService) GetUserActivity(ctx context.Context, userID string) ([]*AuditLog, error) {
	// Simulate user activity retrieval
	time.Sleep(70 * time.Millisecond)
	
	activity := []*AuditLog{
		{
			ID:        fmt.Sprintf("audit_%d", time.Now().Unix()),
			UserID:    userID,
			Action:    "page_view",
			Resource:  "dashboard",
			Details:   map[string]interface{}{"page": "/dashboard"},
			IP:        "192.168.1.1",
			UserAgent: "Mozilla/5.0",
			Timestamp: time.Now().Add(-30 * time.Minute),
		},
		{
			ID:        fmt.Sprintf("audit_%d", time.Now().Unix()-1),
			UserID:    userID,
			Action:    "order_created",
			Resource:  "order",
			Details:   map[string]interface{}{"order_id": "order_123"},
			IP:        "192.168.1.1",
			UserAgent: "Mozilla/5.0",
			Timestamp: time.Now().Add(-1 * time.Hour),
		},
	}
	
	return activity, nil
}
