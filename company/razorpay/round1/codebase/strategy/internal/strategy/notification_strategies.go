package strategy

import (
	"context"
	"fmt"
	"time"
)

// EmailNotificationStrategy implements NotificationStrategy for Email
type EmailNotificationStrategy struct {
	smtpHost  string
	smtpPort  int
	timeout   time.Duration
	available bool
}

// NewEmailNotificationStrategy creates a new Email notification strategy
func NewEmailNotificationStrategy(smtpHost string, smtpPort int, timeout time.Duration) *EmailNotificationStrategy {
	return &EmailNotificationStrategy{
		smtpHost:  smtpHost,
		smtpPort:  smtpPort,
		timeout:   timeout,
		available: true,
	}
}

// SendNotification sends notification via Email
func (e *EmailNotificationStrategy) SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error) {
	// Simulate Email notification sending
	time.Sleep(50 * time.Millisecond)
	
	response := &NotificationResponse{
		NotificationID: request.NotificationID,
		Status:         "sent",
		Channel:        "email",
		SentAt:         time.Now(),
		DeliveryID:     fmt.Sprintf("email_%s", request.NotificationID),
		Metadata:       request.Metadata,
	}
	
	return response, nil
}

// ValidateNotification validates notification request for Email
func (e *EmailNotificationStrategy) ValidateNotification(ctx context.Context, request NotificationRequest) error {
	if request.Title == "" {
		return fmt.Errorf("email title is required")
	}
	if request.Message == "" {
		return fmt.Errorf("email message is required")
	}
	return nil
}

// GetStrategyName returns the strategy name
func (e *EmailNotificationStrategy) GetStrategyName() string {
	return "email"
}

// GetSupportedChannels returns supported channels
func (e *EmailNotificationStrategy) GetSupportedChannels() []string {
	return []string{"email"}
}

// GetDeliveryTime returns delivery time
func (e *EmailNotificationStrategy) GetDeliveryTime() time.Duration {
	return e.timeout
}

// IsAvailable returns availability status
func (e *EmailNotificationStrategy) IsAvailable() bool {
	return e.available
}

// SMSNotificationStrategy implements NotificationStrategy for SMS
type SMSNotificationStrategy struct {
	apiKey    string
	timeout   time.Duration
	available bool
}

// NewSMSNotificationStrategy creates a new SMS notification strategy
func NewSMSNotificationStrategy(apiKey string, timeout time.Duration) *SMSNotificationStrategy {
	return &SMSNotificationStrategy{
		apiKey:    apiKey,
		timeout:   timeout,
		available: true,
	}
}

// SendNotification sends notification via SMS
func (s *SMSNotificationStrategy) SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error) {
	// Simulate SMS notification sending
	time.Sleep(30 * time.Millisecond)
	
	response := &NotificationResponse{
		NotificationID: request.NotificationID,
		Status:         "sent",
		Channel:        "sms",
		SentAt:         time.Now(),
		DeliveryID:     fmt.Sprintf("sms_%s", request.NotificationID),
		Metadata:       request.Metadata,
	}
	
	return response, nil
}

// ValidateNotification validates notification request for SMS
func (s *SMSNotificationStrategy) ValidateNotification(ctx context.Context, request NotificationRequest) error {
	if request.Message == "" {
		return fmt.Errorf("sms message is required")
	}
	if len(request.Message) > 160 {
		return fmt.Errorf("sms message too long: %d characters", len(request.Message))
	}
	return nil
}

// GetStrategyName returns the strategy name
func (s *SMSNotificationStrategy) GetStrategyName() string {
	return "sms"
}

// GetSupportedChannels returns supported channels
func (s *SMSNotificationStrategy) GetSupportedChannels() []string {
	return []string{"sms"}
}

// GetDeliveryTime returns delivery time
func (s *SMSNotificationStrategy) GetDeliveryTime() time.Duration {
	return s.timeout
}

// IsAvailable returns availability status
func (s *SMSNotificationStrategy) IsAvailable() bool {
	return s.available
}

// PushNotificationStrategy implements NotificationStrategy for Push
type PushNotificationStrategy struct {
	apiKey    string
	timeout   time.Duration
	available bool
}

// NewPushNotificationStrategy creates a new Push notification strategy
func NewPushNotificationStrategy(apiKey string, timeout time.Duration) *PushNotificationStrategy {
	return &PushNotificationStrategy{
		apiKey:    apiKey,
		timeout:   timeout,
		available: true,
	}
}

// SendNotification sends notification via Push
func (p *PushNotificationStrategy) SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error) {
	// Simulate Push notification sending
	time.Sleep(20 * time.Millisecond)
	
	response := &NotificationResponse{
		NotificationID: request.NotificationID,
		Status:         "sent",
		Channel:        "push",
		SentAt:         time.Now(),
		DeliveryID:     fmt.Sprintf("push_%s", request.NotificationID),
		Metadata:       request.Metadata,
	}
	
	return response, nil
}

// ValidateNotification validates notification request for Push
func (p *PushNotificationStrategy) ValidateNotification(ctx context.Context, request NotificationRequest) error {
	if request.Title == "" {
		return fmt.Errorf("push title is required")
	}
	if request.Message == "" {
		return fmt.Errorf("push message is required")
	}
	return nil
}

// GetStrategyName returns the strategy name
func (p *PushNotificationStrategy) GetStrategyName() string {
	return "push"
}

// GetSupportedChannels returns supported channels
func (p *PushNotificationStrategy) GetSupportedChannels() []string {
	return []string{"push"}
}

// GetDeliveryTime returns delivery time
func (p *PushNotificationStrategy) GetDeliveryTime() time.Duration {
	return p.timeout
}

// IsAvailable returns availability status
func (p *PushNotificationStrategy) IsAvailable() bool {
	return p.available
}

// WebhookNotificationStrategy implements NotificationStrategy for Webhook
type WebhookNotificationStrategy struct {
	apiKey    string
	timeout   time.Duration
	available bool
}

// NewWebhookNotificationStrategy creates a new Webhook notification strategy
func NewWebhookNotificationStrategy(apiKey string, timeout time.Duration) *WebhookNotificationStrategy {
	return &WebhookNotificationStrategy{
		apiKey:    apiKey,
		timeout:   timeout,
		available: true,
	}
}

// SendNotification sends notification via Webhook
func (w *WebhookNotificationStrategy) SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error) {
	// Simulate Webhook notification sending
	time.Sleep(100 * time.Millisecond)
	
	response := &NotificationResponse{
		NotificationID: request.NotificationID,
		Status:         "sent",
		Channel:        "webhook",
		SentAt:         time.Now(),
		DeliveryID:     fmt.Sprintf("webhook_%s", request.NotificationID),
		Metadata:       request.Metadata,
	}
	
	return response, nil
}

// ValidateNotification validates notification request for Webhook
func (w *WebhookNotificationStrategy) ValidateNotification(ctx context.Context, request NotificationRequest) error {
	if request.Message == "" {
		return fmt.Errorf("webhook message is required")
	}
	return nil
}

// GetStrategyName returns the strategy name
func (w *WebhookNotificationStrategy) GetStrategyName() string {
	return "webhook"
}

// GetSupportedChannels returns supported channels
func (w *WebhookNotificationStrategy) GetSupportedChannels() []string {
	return []string{"webhook"}
}

// GetDeliveryTime returns delivery time
func (w *WebhookNotificationStrategy) GetDeliveryTime() time.Duration {
	return w.timeout
}

// IsAvailable returns availability status
func (w *WebhookNotificationStrategy) IsAvailable() bool {
	return w.available
}

// SlackNotificationStrategy implements NotificationStrategy for Slack
type SlackNotificationStrategy struct {
	apiKey    string
	timeout   time.Duration
	available bool
}

// NewSlackNotificationStrategy creates a new Slack notification strategy
func NewSlackNotificationStrategy(apiKey string, timeout time.Duration) *SlackNotificationStrategy {
	return &SlackNotificationStrategy{
		apiKey:    apiKey,
		timeout:   timeout,
		available: true,
	}
}

// SendNotification sends notification via Slack
func (s *SlackNotificationStrategy) SendNotification(ctx context.Context, request NotificationRequest) (*NotificationResponse, error) {
	// Simulate Slack notification sending
	time.Sleep(80 * time.Millisecond)
	
	response := &NotificationResponse{
		NotificationID: request.NotificationID,
		Status:         "sent",
		Channel:        "slack",
		SentAt:         time.Now(),
		DeliveryID:     fmt.Sprintf("slack_%s", request.NotificationID),
		Metadata:       request.Metadata,
	}
	
	return response, nil
}

// ValidateNotification validates notification request for Slack
func (s *SlackNotificationStrategy) ValidateNotification(ctx context.Context, request NotificationRequest) error {
	if request.Message == "" {
		return fmt.Errorf("slack message is required")
	}
	return nil
}

// GetStrategyName returns the strategy name
func (s *SlackNotificationStrategy) GetStrategyName() string {
	return "slack"
}

// GetSupportedChannels returns supported channels
func (s *SlackNotificationStrategy) GetSupportedChannels() []string {
	return []string{"slack"}
}

// GetDeliveryTime returns delivery time
func (s *SlackNotificationStrategy) GetDeliveryTime() time.Duration {
	return s.timeout
}

// IsAvailable returns availability status
func (s *SlackNotificationStrategy) IsAvailable() bool {
	return s.available
}
