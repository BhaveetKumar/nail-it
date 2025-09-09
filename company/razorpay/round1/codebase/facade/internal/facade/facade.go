package facade

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ECommerceFacade provides a simplified interface to complex e-commerce operations
type ECommerceFacade struct {
	paymentService      PaymentService
	notificationService NotificationService
	userService         UserService
	orderService        OrderService
	inventoryService    InventoryService
	analyticsService    AnalyticsService
	auditService        AuditService
	cacheService        CacheService
	databaseService     DatabaseService
	messageQueueService MessageQueueService
	websocketService    WebSocketService
	securityService     SecurityService
	configService       ConfigurationService
	healthService       HealthService
	monitoringService   MonitoringService
	logger              Logger
	metrics             Metrics
	config              FacadeConfig
	mu                  sync.RWMutex
}

// NewECommerceFacade creates a new e-commerce facade
func NewECommerceFacade(
	paymentService PaymentService,
	notificationService NotificationService,
	userService UserService,
	orderService OrderService,
	inventoryService InventoryService,
	analyticsService AnalyticsService,
	auditService AuditService,
	cacheService CacheService,
	databaseService DatabaseService,
	messageQueueService MessageQueueService,
	websocketService WebSocketService,
	securityService SecurityService,
	configService ConfigurationService,
	healthService HealthService,
	monitoringService MonitoringService,
	logger Logger,
	metrics Metrics,
	config FacadeConfig,
) *ECommerceFacade {
	return &ECommerceFacade{
		paymentService:      paymentService,
		notificationService: notificationService,
		userService:         userService,
		orderService:        orderService,
		inventoryService:    inventoryService,
		analyticsService:    analyticsService,
		auditService:        auditService,
		cacheService:        cacheService,
		databaseService:     databaseService,
		messageQueueService: messageQueueService,
		websocketService:    websocketService,
		securityService:     securityService,
		configService:       configService,
		healthService:       healthService,
		monitoringService:   monitoringService,
		logger:              logger,
		metrics:             metrics,
		config:              config,
	}
}

// ProcessOrder processes a complete order with all necessary steps
func (f *ECommerceFacade) ProcessOrder(ctx context.Context, request ProcessOrderRequest) (*ProcessOrderResponse, error) {
	start := time.Now()
	
	// Validate user authentication
	tokenClaims, err := f.securityService.ValidateToken(ctx, request.Token)
	if err != nil {
		f.logger.Error("Authentication failed", "error", err)
		f.metrics.IncrementCounter("facade_auth_failures", map[string]string{"operation": "process_order"})
		return nil, fmt.Errorf("authentication failed: %w", err)
	}
	
	// Check inventory availability
	for _, item := range request.Items {
		available, err := f.inventoryService.CheckAvailability(ctx, item.ProductID, item.Quantity)
		if err != nil {
			f.logger.Error("Inventory check failed", "product_id", item.ProductID, "error", err)
			return nil, fmt.Errorf("inventory check failed: %w", err)
		}
		if !available {
			return nil, fmt.Errorf("product %s not available in requested quantity", item.ProductID)
		}
	}
	
	// Reserve products
	for _, item := range request.Items {
		if err := f.inventoryService.ReserveProduct(ctx, item.ProductID, item.Quantity); err != nil {
			f.logger.Error("Product reservation failed", "product_id", item.ProductID, "error", err)
			// Release previously reserved products
			f.releaseReservedProducts(ctx, request.Items[:len(request.Items)-1])
			return nil, fmt.Errorf("product reservation failed: %w", err)
		}
	}
	
	// Create order
	order, err := f.orderService.CreateOrder(ctx, CreateOrderRequest{
		UserID:          tokenClaims.UserID,
		Items:           request.Items,
		ShippingAddress: request.ShippingAddress,
		BillingAddress:  request.BillingAddress,
		Metadata:        request.Metadata,
	})
	if err != nil {
		f.logger.Error("Order creation failed", "error", err)
		f.releaseReservedProducts(ctx, request.Items)
		return nil, fmt.Errorf("order creation failed: %w", err)
	}
	
	// Process payment
	paymentResponse, err := f.paymentService.ProcessPayment(ctx, PaymentRequest{
		ID:            order.ID,
		Amount:        order.Total,
		Currency:      "INR",
		CustomerID:    tokenClaims.UserID,
		MerchantID:    "merchant_123",
		Description:   fmt.Sprintf("Order %s", order.ID),
		PaymentMethod: request.PaymentMethod,
		Metadata:      request.Metadata,
		CreatedAt:     time.Now(),
	})
	if err != nil {
		f.logger.Error("Payment processing failed", "order_id", order.ID, "error", err)
		f.orderService.CancelOrder(ctx, order.ID)
		f.releaseReservedProducts(ctx, request.Items)
		return nil, fmt.Errorf("payment processing failed: %w", err)
	}
	
	// Update order with payment information
	order.Status = "paid"
	order.Metadata["payment_id"] = paymentResponse.ID
	order.Metadata["transaction_id"] = paymentResponse.TransactionID
	
	// Send notifications
	go f.sendOrderNotifications(ctx, order, paymentResponse)
	
	// Track analytics
	go f.trackOrderAnalytics(ctx, order, tokenClaims.UserID)
	
	// Audit log
	go f.auditService.LogAction(ctx, AuditAction{
		ID:        fmt.Sprintf("audit_%d", time.Now().Unix()),
		UserID:    tokenClaims.UserID,
		Action:    "order_created",
		Resource:  "order",
		Details:   map[string]interface{}{"order_id": order.ID, "amount": order.Total},
		IP:        request.IP,
		UserAgent: request.UserAgent,
		Timestamp: time.Now(),
	})
	
	duration := time.Since(start)
	f.metrics.RecordTiming("facade_process_order_duration", duration, map[string]string{"status": "success"})
	f.metrics.IncrementCounter("facade_orders_processed", map[string]string{"status": "success"})
	
	return &ProcessOrderResponse{
		Order:   order,
		Payment: paymentResponse,
		Message: "Order processed successfully",
	}, nil
}

// GetUserDashboard provides a comprehensive user dashboard
func (f *ECommerceFacade) GetUserDashboard(ctx context.Context, userID string) (*UserDashboard, error) {
	start := time.Now()
	
	// Get user information
	user, err := f.userService.GetUser(ctx, userID)
	if err != nil {
		f.logger.Error("Failed to get user", "user_id", userID, "error", err)
		return nil, fmt.Errorf("failed to get user: %w", err)
	}
	
	// Get order history
	orders, err := f.orderService.GetOrderHistory(ctx, userID)
	if err != nil {
		f.logger.Error("Failed to get order history", "user_id", userID, "error", err)
		orders = []*Order{} // Continue with empty orders
	}
	
	// Get payment history
	payments, err := f.paymentService.GetPaymentHistory(ctx, userID)
	if err != nil {
		f.logger.Error("Failed to get payment history", "user_id", userID, "error", err)
		payments = []*PaymentHistory{} // Continue with empty payments
	}
	
	// Get user analytics
	analytics, err := f.analyticsService.GetUserAnalytics(ctx, userID)
	if err != nil {
		f.logger.Error("Failed to get user analytics", "user_id", userID, "error", err)
		analytics = &UserAnalytics{UserID: userID} // Continue with empty analytics
	}
	
	// Get recent activity
	activity, err := f.auditService.GetUserActivity(ctx, userID)
	if err != nil {
		f.logger.Error("Failed to get user activity", "user_id", userID, "error", err)
		activity = []*AuditLog{} // Continue with empty activity
	}
	
	duration := time.Since(start)
	f.metrics.RecordTiming("facade_user_dashboard_duration", duration, map[string]string{"status": "success"})
	f.metrics.IncrementCounter("facade_dashboards_retrieved", map[string]string{"status": "success"})
	
	return &UserDashboard{
		User:     user,
		Orders:   orders,
		Payments: payments,
		Analytics: analytics,
		Activity: activity,
		LastUpdated: time.Now(),
	}, nil
}

// SendNotification sends notifications through multiple channels
func (f *ECommerceFacade) SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error) {
	start := time.Now()
	
	var errors []error
	var sentChannels []string
	
	// Send email if requested
	if request.Email != nil {
		if err := f.notificationService.SendEmail(ctx, *request.Email); err != nil {
			f.logger.Error("Email sending failed", "error", err)
			errors = append(errors, fmt.Errorf("email: %w", err))
		} else {
			sentChannels = append(sentChannels, "email")
		}
	}
	
	// Send SMS if requested
	if request.SMS != nil {
		if err := f.notificationService.SendSMS(ctx, *request.SMS); err != nil {
			f.logger.Error("SMS sending failed", "error", err)
			errors = append(errors, fmt.Errorf("sms: %w", err))
		} else {
			sentChannels = append(sentChannels, "sms")
		}
	}
	
	// Send push notification if requested
	if request.Push != nil {
		if err := f.notificationService.SendPushNotification(ctx, *request.Push); err != nil {
			f.logger.Error("Push notification sending failed", "error", err)
			errors = append(errors, fmt.Errorf("push: %w", err))
		} else {
			sentChannels = append(sentChannels, "push")
		}
	}
	
	duration := time.Since(start)
	f.metrics.RecordTiming("facade_notification_duration", duration, map[string]string{"channels": fmt.Sprintf("%d", len(sentChannels))})
	f.metrics.IncrementCounter("facade_notifications_sent", map[string]string{"channels": fmt.Sprintf("%d", len(sentChannels))})
	
	response := &NotificationResponse{
		SentChannels: sentChannels,
		Errors:       errors,
		Message:      fmt.Sprintf("Notifications sent to %d channels", len(sentChannels)),
		Timestamp:    time.Now(),
	}
	
	// Return partial success if some channels failed
	if len(errors) > 0 && len(sentChannels) == 0 {
		return nil, fmt.Errorf("all notification channels failed: %v", errors)
	}
	
	return response, nil
}

// GetSystemHealth provides comprehensive system health information
func (f *ECommerceFacade) GetSystemHealth(ctx context.Context) (*SystemHealth, error) {
	start := time.Now()
	
	// Get overall health status
	healthStatus, err := f.healthService.CheckHealth(ctx)
	if err != nil {
		f.logger.Error("Failed to get health status", "error", err)
		return nil, fmt.Errorf("failed to get health status: %w", err)
	}
	
	// Get system metrics
	systemMetrics, err := f.monitoringService.GetSystemMetrics(ctx)
	if err != nil {
		f.logger.Error("Failed to get system metrics", "error", err)
		systemMetrics = &SystemMetrics{Timestamp: time.Now()} // Continue with empty metrics
	}
	
	// Get service metrics
	serviceMetrics := make(map[string]*ServiceMetrics)
	services := []string{"payment", "notification", "user", "order", "inventory", "analytics", "audit"}
	
	for _, service := range services {
		metrics, err := f.monitoringService.GetServiceMetrics(ctx, service)
		if err != nil {
			f.logger.Error("Failed to get service metrics", "service", service, "error", err)
			continue
		}
		serviceMetrics[service] = metrics
	}
	
	duration := time.Since(start)
	f.metrics.RecordTiming("facade_system_health_duration", duration, map[string]string{"status": "success"})
	f.metrics.IncrementCounter("facade_health_checks", map[string]string{"status": "success"})
	
	return &SystemHealth{
		Overall:         healthStatus,
		SystemMetrics:   systemMetrics,
		ServiceMetrics:  serviceMetrics,
		LastChecked:     time.Now(),
		ResponseTime:    duration,
	}, nil
}

// Helper methods

// releaseReservedProducts releases previously reserved products
func (f *ECommerceFacade) releaseReservedProducts(ctx context.Context, items []OrderItem) {
	for _, item := range items {
		if err := f.inventoryService.ReleaseProduct(ctx, item.ProductID, item.Quantity); err != nil {
			f.logger.Error("Failed to release product", "product_id", item.ProductID, "error", err)
		}
	}
}

// sendOrderNotifications sends order-related notifications
func (f *ECommerceFacade) sendOrderNotifications(ctx context.Context, order *Order, payment *PaymentResponse) {
	// Send order confirmation email
	emailRequest := EmailRequest{
		To:      order.Metadata["email"].(string),
		Subject: "Order Confirmation",
		Body:    fmt.Sprintf("Your order %s has been confirmed. Payment ID: %s", order.ID, payment.ID),
		Type:    "order_confirmation",
		Data:    map[string]interface{}{"order": order, "payment": payment},
	}
	
	if err := f.notificationService.SendEmail(ctx, emailRequest); err != nil {
		f.logger.Error("Failed to send order confirmation email", "order_id", order.ID, "error", err)
	}
	
	// Send SMS notification
	smsRequest := SMSRequest{
		To:      order.Metadata["phone"].(string),
		Message: fmt.Sprintf("Order %s confirmed. Amount: ₹%.2f", order.ID, order.Total),
		Type:    "order_confirmation",
		Data:    map[string]interface{}{"order": order},
	}
	
	if err := f.notificationService.SendSMS(ctx, smsRequest); err != nil {
		f.logger.Error("Failed to send order confirmation SMS", "order_id", order.ID, "error", err)
	}
}

// trackOrderAnalytics tracks order-related analytics
func (f *ECommerceFacade) trackOrderAnalytics(ctx context.Context, order *Order, userID string) {
	// Track order creation event
	event := Event{
		ID:        fmt.Sprintf("event_%d", time.Now().Unix()),
		UserID:    userID,
		EventType: "order_created",
		Properties: map[string]interface{}{
			"order_id": order.ID,
			"amount":   order.Total,
			"items":    len(order.Items),
		},
		Timestamp: time.Now(),
	}
	
	if err := f.analyticsService.TrackEvent(ctx, event); err != nil {
		f.logger.Error("Failed to track order analytics", "order_id", order.ID, "error", err)
	}
	
	// Track product views for each item
	for _, item := range order.Items {
		productEvent := Event{
			ID:        fmt.Sprintf("event_%d", time.Now().Unix()),
			UserID:    userID,
			EventType: "product_purchased",
			Properties: map[string]interface{}{
				"product_id": item.ProductID,
				"quantity":   item.Quantity,
				"price":      item.Price,
			},
			Timestamp: time.Now(),
		}
		
		if err := f.analyticsService.TrackEvent(ctx, productEvent); err != nil {
			f.logger.Error("Failed to track product analytics", "product_id", item.ProductID, "error", err)
		}
	}
}

// ProcessOrderRequest represents a process order request
type ProcessOrderRequest struct {
	Token           string                 `json:"token"`
	Items           []OrderItem            `json:"items"`
	ShippingAddress map[string]interface{} `json:"shipping_address"`
	BillingAddress  map[string]interface{} `json:"billing_address"`
	PaymentMethod   string                 `json:"payment_method"`
	Metadata        map[string]interface{} `json:"metadata"`
	IP              string                 `json:"ip"`
	UserAgent       string                 `json:"user_agent"`
}

// ProcessOrderResponse represents a process order response
type ProcessOrderResponse struct {
	Order   *Order          `json:"order"`
	Payment *PaymentResponse `json:"payment"`
	Message string          `json:"message"`
}

// UserDashboard represents a user dashboard
type UserDashboard struct {
	User       *User             `json:"user"`
	Orders     []*Order          `json:"orders"`
	Payments   []*PaymentHistory `json:"payments"`
	Analytics  *UserAnalytics    `json:"analytics"`
	Activity   []*AuditLog       `json:"activity"`
	LastUpdated time.Time        `json:"last_updated"`
}

// NotificationRequest represents a notification request
type NotificationRequest struct {
	Email *EmailRequest `json:"email,omitempty"`
	SMS   *SMSRequest   `json:"sms,omitempty"`
	Push  *PushRequest  `json:"push,omitempty"`
}

// NotificationResponse represents a notification response
type NotificationResponse struct {
	SentChannels []string  `json:"sent_channels"`
	Errors       []error   `json:"errors,omitempty"`
	Message      string    `json:"message"`
	Timestamp    time.Time `json:"timestamp"`
}

// SystemHealth represents system health information
type SystemHealth struct {
	Overall        *HealthStatus              `json:"overall"`
	SystemMetrics  *SystemMetrics             `json:"system_metrics"`
	ServiceMetrics map[string]*ServiceMetrics `json:"service_metrics"`
	LastChecked    time.Time                  `json:"last_checked"`
	ResponseTime   time.Duration              `json:"response_time"`
}
