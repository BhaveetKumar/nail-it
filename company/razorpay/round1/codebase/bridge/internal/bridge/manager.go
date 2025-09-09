package bridge

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// PaymentManager manages payment processing using different gateways
type PaymentManager struct {
	gateways map[string]PaymentGateway
	metrics  *PaymentMetrics
	mu       sync.RWMutex
}

// NewPaymentManager creates a new payment manager
func NewPaymentManager() *PaymentManager {
	return &PaymentManager{
		gateways: make(map[string]PaymentGateway),
		metrics:  &PaymentMetrics{},
	}
}

// RegisterGateway registers a payment gateway
func (pm *PaymentManager) RegisterGateway(name string, gateway PaymentGateway) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.gateways[name] = gateway
}

// ProcessPayment processes a payment using the specified gateway
func (pm *PaymentManager) ProcessPayment(ctx context.Context, gatewayName string, req PaymentRequest) (*PaymentResponse, error) {
	pm.mu.RLock()
	gateway, exists := pm.gateways[gatewayName]
	pm.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("gateway %s not found", gatewayName)
	}
	
	// Update metrics
	pm.mu.Lock()
	pm.metrics.TotalProcessed++
	pm.metrics.TotalAmount += req.Amount
	pm.metrics.LastProcessed = time.Now()
	pm.mu.Unlock()
	
	// Process payment
	response, err := gateway.ProcessPayment(ctx, req)
	if err != nil {
		pm.mu.Lock()
		pm.metrics.FailedPayments++
		pm.mu.Unlock()
		return nil, err
	}
	
	// Update success metrics
	pm.mu.Lock()
	pm.metrics.SuccessfulPayments++
	pm.metrics.AverageAmount = pm.metrics.TotalAmount / float64(pm.metrics.TotalProcessed)
	if pm.metrics.TotalProcessed > 0 {
		pm.metrics.SuccessRate = float64(pm.metrics.SuccessfulPayments) / float64(pm.metrics.TotalProcessed) * 100
	}
	pm.mu.Unlock()
	
	return response, nil
}

// RefundPayment refunds a payment using the specified gateway
func (pm *PaymentManager) RefundPayment(ctx context.Context, gatewayName string, transactionID string, amount float64) error {
	pm.mu.RLock()
	gateway, exists := pm.gateways[gatewayName]
	pm.mu.RUnlock()
	
	if !exists {
		return fmt.Errorf("gateway %s not found", gatewayName)
	}
	
	return gateway.RefundPayment(ctx, transactionID, amount)
}

// GetMetrics returns payment metrics
func (pm *PaymentManager) GetMetrics() PaymentMetrics {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	return *pm.metrics
}

// NotificationManager manages notification sending using different channels
type NotificationManager struct {
	channels map[string]NotificationChannel
	metrics  *NotificationMetrics
	mu       sync.RWMutex
}

// NewNotificationManager creates a new notification manager
func NewNotificationManager() *NotificationManager {
	return &NotificationManager{
		channels: make(map[string]NotificationChannel),
		metrics:  &NotificationMetrics{},
	}
}

// RegisterChannel registers a notification channel
func (nm *NotificationManager) RegisterChannel(name string, channel NotificationChannel) {
	nm.mu.Lock()
	defer nm.mu.Unlock()
	nm.channels[name] = channel
}

// SendNotification sends a notification using the specified channel
func (nm *NotificationManager) SendNotification(ctx context.Context, channelName string, req NotificationRequest) (*NotificationResponse, error) {
	nm.mu.RLock()
	channel, exists := nm.channels[channelName]
	nm.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("channel %s not found", channelName)
	}
	
	// Update metrics
	nm.mu.Lock()
	nm.metrics.TotalSent++
	nm.metrics.LastSent = time.Now()
	nm.mu.Unlock()
	
	// Send notification
	response, err := channel.SendNotification(ctx, req)
	if err != nil {
		nm.mu.Lock()
		nm.metrics.FailedSends++
		nm.mu.Unlock()
		return nil, err
	}
	
	// Update success metrics
	nm.mu.Lock()
	nm.metrics.SuccessfulSends++
	if nm.metrics.TotalSent > 0 {
		nm.metrics.SuccessRate = float64(nm.metrics.SuccessfulSends) / float64(nm.metrics.TotalSent) * 100
	}
	nm.mu.Unlock()
	
	return response, nil
}

// GetMetrics returns notification metrics
func (nm *NotificationManager) GetMetrics() NotificationMetrics {
	nm.mu.RLock()
	defer nm.mu.RUnlock()
	return *nm.metrics
}

// BridgeService combines payment and notification services
type BridgeService struct {
	paymentManager      *PaymentManager
	notificationManager *NotificationManager
}

// NewBridgeService creates a new bridge service
func NewBridgeService() *BridgeService {
	return &BridgeService{
		paymentManager:      NewPaymentManager(),
		notificationManager: NewNotificationManager(),
	}
}

// ProcessPaymentWithNotification processes a payment and sends notification
func (bs *BridgeService) ProcessPaymentWithNotification(ctx context.Context, gatewayName string, channelName string, paymentReq PaymentRequest, notificationReq NotificationRequest) (*PaymentResponse, *NotificationResponse, error) {
	// Process payment
	paymentResp, err := bs.paymentManager.ProcessPayment(ctx, gatewayName, paymentReq)
	if err != nil {
		return nil, nil, fmt.Errorf("payment processing failed: %w", err)
	}
	
	// Update notification request with payment details
	notificationReq.Metadata["payment_id"] = paymentResp.ID
	notificationReq.Metadata["transaction_id"] = paymentResp.TransactionID
	notificationReq.Metadata["amount"] = paymentResp.Amount
	notificationReq.Metadata["currency"] = paymentResp.Currency
	
	// Send notification
	notificationResp, err := bs.notificationManager.SendNotification(ctx, channelName, notificationReq)
	if err != nil {
		// Payment succeeded but notification failed - log but don't fail the transaction
		return paymentResp, nil, fmt.Errorf("notification sending failed: %w", err)
	}
	
	return paymentResp, notificationResp, nil
}

// GetPaymentMetrics returns payment metrics
func (bs *BridgeService) GetPaymentMetrics() PaymentMetrics {
	return bs.paymentManager.GetMetrics()
}

// GetNotificationMetrics returns notification metrics
func (bs *BridgeService) GetNotificationMetrics() NotificationMetrics {
	return bs.notificationManager.GetMetrics()
}
