package bridge

import (
	"context"
	"sync"
	"time"
)

// MetricsCollector collects and aggregates metrics
type MetricsCollector struct {
	paymentMetrics      *PaymentMetrics
	notificationMetrics *NotificationMetrics
	mu                  sync.RWMutex
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		paymentMetrics:      &PaymentMetrics{},
		notificationMetrics: &NotificationMetrics{},
	}
}

// RecordPaymentSuccess records a successful payment
func (mc *MetricsCollector) RecordPaymentSuccess(amount float64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	
	mc.paymentMetrics.TotalProcessed++
	mc.paymentMetrics.SuccessfulPayments++
	mc.paymentMetrics.TotalAmount += amount
	mc.paymentMetrics.LastProcessed = time.Now()
	
	if mc.paymentMetrics.TotalProcessed > 0 {
		mc.paymentMetrics.AverageAmount = mc.paymentMetrics.TotalAmount / float64(mc.paymentMetrics.TotalProcessed)
		mc.paymentMetrics.SuccessRate = float64(mc.paymentMetrics.SuccessfulPayments) / float64(mc.paymentMetrics.TotalProcessed) * 100
	}
}

// RecordPaymentFailure records a failed payment
func (mc *MetricsCollector) RecordPaymentFailure() {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	
	mc.paymentMetrics.TotalProcessed++
	mc.paymentMetrics.FailedPayments++
	mc.paymentMetrics.LastProcessed = time.Now()
	
	if mc.paymentMetrics.TotalProcessed > 0 {
		mc.paymentMetrics.SuccessRate = float64(mc.paymentMetrics.SuccessfulPayments) / float64(mc.paymentMetrics.TotalProcessed) * 100
	}
}

// RecordNotificationSuccess records a successful notification
func (mc *MetricsCollector) RecordNotificationSuccess() {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	
	mc.notificationMetrics.TotalSent++
	mc.notificationMetrics.SuccessfulSends++
	mc.notificationMetrics.LastSent = time.Now()
	
	if mc.notificationMetrics.TotalSent > 0 {
		mc.notificationMetrics.SuccessRate = float64(mc.notificationMetrics.SuccessfulSends) / float64(mc.notificationMetrics.TotalSent) * 100
	}
}

// RecordNotificationFailure records a failed notification
func (mc *MetricsCollector) RecordNotificationFailure() {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	
	mc.notificationMetrics.TotalSent++
	mc.notificationMetrics.FailedSends++
	mc.notificationMetrics.LastSent = time.Now()
	
	if mc.notificationMetrics.TotalSent > 0 {
		mc.notificationMetrics.SuccessRate = float64(mc.notificationMetrics.SuccessfulSends) / float64(mc.notificationMetrics.TotalSent) * 100
	}
}

// GetPaymentMetrics returns payment metrics
func (mc *MetricsCollector) GetPaymentMetrics() PaymentMetrics {
	mc.mu.RLock()
	defer mc.mu.RUnlock()
	return *mc.paymentMetrics
}

// GetNotificationMetrics returns notification metrics
func (mc *MetricsCollector) GetNotificationMetrics() NotificationMetrics {
	mc.mu.RLock()
	defer mc.mu.RUnlock()
	return *mc.notificationMetrics
}

// ResetMetrics resets all metrics
func (mc *MetricsCollector) ResetMetrics() {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	
	mc.paymentMetrics = &PaymentMetrics{}
	mc.notificationMetrics = &NotificationMetrics{}
}

// MetricsService provides metrics endpoints
type MetricsService struct {
	collector *MetricsCollector
}

// NewMetricsService creates a new metrics service
func NewMetricsService() *MetricsService {
	return &MetricsService{
		collector: NewMetricsCollector(),
	}
}

// GetPaymentMetrics returns payment metrics
func (ms *MetricsService) GetPaymentMetrics(ctx context.Context) (*PaymentMetrics, error) {
	metrics := ms.collector.GetPaymentMetrics()
	return &metrics, nil
}

// GetNotificationMetrics returns notification metrics
func (ms *MetricsService) GetNotificationMetrics(ctx context.Context) (*NotificationMetrics, error) {
	metrics := ms.collector.GetNotificationMetrics()
	return &metrics, nil
}

// GetHealthStatus returns health status
func (ms *MetricsService) GetHealthStatus(ctx context.Context) map[string]interface{} {
	paymentMetrics := ms.collector.GetPaymentMetrics()
	notificationMetrics := ms.collector.GetNotificationMetrics()
	
	return map[string]interface{}{
		"status": "healthy",
		"timestamp": time.Now(),
		"payment_metrics": paymentMetrics,
		"notification_metrics": notificationMetrics,
	}
}

// RecordPaymentSuccess records a successful payment
func (ms *MetricsService) RecordPaymentSuccess(amount float64) {
	ms.collector.RecordPaymentSuccess(amount)
}

// RecordPaymentFailure records a failed payment
func (ms *MetricsService) RecordPaymentFailure() {
	ms.collector.RecordPaymentFailure()
}

// RecordNotificationSuccess records a successful notification
func (ms *MetricsService) RecordNotificationSuccess() {
	ms.collector.RecordNotificationSuccess()
}

// RecordNotificationFailure records a failed notification
func (ms *MetricsService) RecordNotificationFailure() {
	ms.collector.RecordNotificationFailure()
}
