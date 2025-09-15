package adapter

import (
	"context"
	"fmt"
	"time"
)

// EmailNotificationAdapter adapts Email notification service
type EmailNotificationAdapter struct {
	smtpHost  string
	smtpPort  int
	timeout   time.Duration
	available bool
}

// NewEmailNotificationAdapter creates a new Email notification adapter
func NewEmailNotificationAdapter(smtpHost string, smtpPort int, timeout time.Duration) *EmailNotificationAdapter {
	return &EmailNotificationAdapter{
		smtpHost:  smtpHost,
		smtpPort:  smtpPort,
		timeout:   timeout,
		available: true,
	}
}

// SendNotification sends notification via Email
func (e *EmailNotificationAdapter) SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error) {
	// Simulate Email notification sending
	time.Sleep(50 * time.Millisecond)

	response := &NotificationResponse{
		NotificationID: request.NotificationID,
		Status:         "sent",
		Channel:        "email",
		Type:           request.Type,
		SentAt:         time.Now(),
		DeliveryID:     fmt.Sprintf("email_%s", request.NotificationID),
		Metadata:       request.Metadata,
	}

	return response, nil
}

// GetNotificationStatus gets notification status from Email service
func (e *EmailNotificationAdapter) GetNotificationStatus(ctx context.Context, notificationID string) (*NotificationStatus, error) {
	// Simulate Email status check
	time.Sleep(30 * time.Millisecond)

	status := &NotificationStatus{
		NotificationID: notificationID,
		Status:         "delivered",
		Channel:        "email",
		Type:           "payment",
		LastUpdated:    time.Now(),
		Metadata:       make(map[string]string),
	}

	return status, nil
}

// GetServiceName returns the service name
func (e *EmailNotificationAdapter) GetServiceName() string {
	return "email"
}

// IsAvailable returns availability status
func (e *EmailNotificationAdapter) IsAvailable() bool {
	return e.available
}

// SMSNotificationAdapter adapts SMS notification service
type SMSNotificationAdapter struct {
	apiKey    string
	timeout   time.Duration
	available bool
}

// NewSMSNotificationAdapter creates a new SMS notification adapter
func NewSMSNotificationAdapter(apiKey string, timeout time.Duration) *SMSNotificationAdapter {
	return &SMSNotificationAdapter{
		apiKey:    apiKey,
		timeout:   timeout,
		available: true,
	}
}

// SendNotification sends notification via SMS
func (s *SMSNotificationAdapter) SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error) {
	// Simulate SMS notification sending
	time.Sleep(30 * time.Millisecond)

	response := &NotificationResponse{
		NotificationID: request.NotificationID,
		Status:         "sent",
		Channel:        "sms",
		Type:           request.Type,
		SentAt:         time.Now(),
		DeliveryID:     fmt.Sprintf("sms_%s", request.NotificationID),
		Metadata:       request.Metadata,
	}

	return response, nil
}

// GetNotificationStatus gets notification status from SMS service
func (s *SMSNotificationAdapter) GetNotificationStatus(ctx context.Context, notificationID string) (*NotificationStatus, error) {
	// Simulate SMS status check
	time.Sleep(20 * time.Millisecond)

	status := &NotificationStatus{
		NotificationID: notificationID,
		Status:         "delivered",
		Channel:        "sms",
		Type:           "payment",
		LastUpdated:    time.Now(),
		Metadata:       make(map[string]string),
	}

	return status, nil
}

// GetServiceName returns the service name
func (s *SMSNotificationAdapter) GetServiceName() string {
	return "sms"
}

// IsAvailable returns availability status
func (s *SMSNotificationAdapter) IsAvailable() bool {
	return s.available
}

// PushNotificationAdapter adapts Push notification service
type PushNotificationAdapter struct {
	apiKey    string
	timeout   time.Duration
	available bool
}

// NewPushNotificationAdapter creates a new Push notification adapter
func NewPushNotificationAdapter(apiKey string, timeout time.Duration) *PushNotificationAdapter {
	return &PushNotificationAdapter{
		apiKey:    apiKey,
		timeout:   timeout,
		available: true,
	}
}

// SendNotification sends notification via Push
func (p *PushNotificationAdapter) SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error) {
	// Simulate Push notification sending
	time.Sleep(20 * time.Millisecond)

	response := &NotificationResponse{
		NotificationID: request.NotificationID,
		Status:         "sent",
		Channel:        "push",
		Type:           request.Type,
		SentAt:         time.Now(),
		DeliveryID:     fmt.Sprintf("push_%s", request.NotificationID),
		Metadata:       request.Metadata,
	}

	return response, nil
}

// GetNotificationStatus gets notification status from Push service
func (p *PushNotificationAdapter) GetNotificationStatus(ctx context.Context, notificationID string) (*NotificationStatus, error) {
	// Simulate Push status check
	time.Sleep(15 * time.Millisecond)

	status := &NotificationStatus{
		NotificationID: notificationID,
		Status:         "delivered",
		Channel:        "push",
		Type:           "payment",
		LastUpdated:    time.Now(),
		Metadata:       make(map[string]string),
	}

	return status, nil
}

// GetServiceName returns the service name
func (p *PushNotificationAdapter) GetServiceName() string {
	return "push"
}

// IsAvailable returns availability status
func (p *PushNotificationAdapter) IsAvailable() bool {
	return p.available
}

// WebhookNotificationAdapter adapts Webhook notification service
type WebhookNotificationAdapter struct {
	apiKey    string
	timeout   time.Duration
	available bool
}

// NewWebhookNotificationAdapter creates a new Webhook notification adapter
func NewWebhookNotificationAdapter(apiKey string, timeout time.Duration) *WebhookNotificationAdapter {
	return &WebhookNotificationAdapter{
		apiKey:    apiKey,
		timeout:   timeout,
		available: true,
	}
}

// SendNotification sends notification via Webhook
func (w *WebhookNotificationAdapter) SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error) {
	// Simulate Webhook notification sending
	time.Sleep(100 * time.Millisecond)

	response := &NotificationResponse{
		NotificationID: request.NotificationID,
		Status:         "sent",
		Channel:        "webhook",
		Type:           request.Type,
		SentAt:         time.Now(),
		DeliveryID:     fmt.Sprintf("webhook_%s", request.NotificationID),
		Metadata:       request.Metadata,
	}

	return response, nil
}

// GetNotificationStatus gets notification status from Webhook service
func (w *WebhookNotificationAdapter) GetNotificationStatus(ctx context.Context, notificationID string) (*NotificationStatus, error) {
	// Simulate Webhook status check
	time.Sleep(50 * time.Millisecond)

	status := &NotificationStatus{
		NotificationID: notificationID,
		Status:         "delivered",
		Channel:        "webhook",
		Type:           "payment",
		LastUpdated:    time.Now(),
		Metadata:       make(map[string]string),
	}

	return status, nil
}

// GetServiceName returns the service name
func (w *WebhookNotificationAdapter) GetServiceName() string {
	return "webhook"
}

// IsAvailable returns availability status
func (w *WebhookNotificationAdapter) IsAvailable() bool {
	return w.available
}

// SlackNotificationAdapter adapts Slack notification service
type SlackNotificationAdapter struct {
	apiKey    string
	timeout   time.Duration
	available bool
}

// NewSlackNotificationAdapter creates a new Slack notification adapter
func NewSlackNotificationAdapter(apiKey string, timeout time.Duration) *SlackNotificationAdapter {
	return &SlackNotificationAdapter{
		apiKey:    apiKey,
		timeout:   timeout,
		available: true,
	}
}

// SendNotification sends notification via Slack
func (s *SlackNotificationAdapter) SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error) {
	// Simulate Slack notification sending
	time.Sleep(80 * time.Millisecond)

	response := &NotificationResponse{
		NotificationID: request.NotificationID,
		Status:         "sent",
		Channel:        "slack",
		Type:           request.Type,
		SentAt:         time.Now(),
		DeliveryID:     fmt.Sprintf("slack_%s", request.NotificationID),
		Metadata:       request.Metadata,
	}

	return response, nil
}

// GetNotificationStatus gets notification status from Slack service
func (s *SlackNotificationAdapter) GetNotificationStatus(ctx context.Context, notificationID string) (*NotificationStatus, error) {
	// Simulate Slack status check
	time.Sleep(40 * time.Millisecond)

	status := &NotificationStatus{
		NotificationID: notificationID,
		Status:         "delivered",
		Channel:        "slack",
		Type:           "payment",
		LastUpdated:    time.Now(),
		Metadata:       make(map[string]string),
	}

	return status, nil
}

// GetServiceName returns the service name
func (s *SlackNotificationAdapter) GetServiceName() string {
	return "slack"
}

// IsAvailable returns availability status
func (s *SlackNotificationAdapter) IsAvailable() bool {
	return s.available
}
