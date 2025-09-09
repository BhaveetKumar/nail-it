package factory

import (
	"context"
	"fmt"
	"sync"

	"factory-service/internal/config"
	"factory-service/internal/logger"
	"factory-service/internal/models"
)

// NotificationChannel interface defines the contract for notification channels
type NotificationChannel interface {
	SendNotification(ctx context.Context, request *models.NotificationRequest) (*models.NotificationResponse, error)
	ValidateNotification(request *models.NotificationRequest) error
	GetChannelType() string
	GetChannelName() string
}

// NotificationChannelFactory implements the Factory pattern for creating notification channels
type NotificationChannelFactory struct {
	channels map[string]func() NotificationChannel
	mutex    sync.RWMutex
}

var (
	notificationChannelFactory *NotificationChannelFactory
	notificationFactoryOnce    sync.Once
)

// GetNotificationChannelFactory returns the singleton instance of NotificationChannelFactory
func GetNotificationChannelFactory() *NotificationChannelFactory {
	notificationFactoryOnce.Do(func() {
		notificationChannelFactory = &NotificationChannelFactory{
			channels: make(map[string]func() NotificationChannel),
		}
		notificationChannelFactory.registerDefaultChannels()
	})
	return notificationChannelFactory
}

// registerDefaultChannels registers the default notification channels
func (ncf *NotificationChannelFactory) registerDefaultChannels() {
	ncf.mutex.Lock()
	defer ncf.mutex.Unlock()

	// Register Email channel
	ncf.channels["email"] = func() NotificationChannel {
		return NewEmailChannel()
	}

	// Register SMS channel
	ncf.channels["sms"] = func() NotificationChannel {
		return NewSMSChannel()
	}

	// Register Push channel
	ncf.channels["push"] = func() NotificationChannel {
		return NewPushChannel()
	}

	// Register WhatsApp channel
	ncf.channels["whatsapp"] = func() NotificationChannel {
		return NewWhatsAppChannel()
	}

	// Register Slack channel
	ncf.channels["slack"] = func() NotificationChannel {
		return NewSlackChannel()
	}
}

// RegisterChannel registers a new notification channel
func (ncf *NotificationChannelFactory) RegisterChannel(name string, creator func() NotificationChannel) {
	ncf.mutex.Lock()
	defer ncf.mutex.Unlock()
	ncf.channels[name] = creator
}

// CreateChannel creates a notification channel instance
func (ncf *NotificationChannelFactory) CreateChannel(channelType string) (NotificationChannel, error) {
	ncf.mutex.RLock()
	creator, exists := ncf.channels[channelType]
	ncf.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("notification channel type '%s' not supported", channelType)
	}

	return creator(), nil
}

// GetAvailableChannels returns the list of available channel types
func (ncf *NotificationChannelFactory) GetAvailableChannels() []string {
	ncf.mutex.RLock()
	defer ncf.mutex.RUnlock()

	channels := make([]string, 0, len(ncf.channels))
	for name := range ncf.channels {
		channels = append(channels, name)
	}
	return channels
}

// EmailChannel implements NotificationChannel for Email
type EmailChannel struct {
	smtpHost string
	smtpPort int
	username string
	password string
}

// NewEmailChannel creates a new Email channel instance
func NewEmailChannel() *EmailChannel {
	cfg := config.GetConfigManager()
	emailConfig := cfg.GetEmailConfig()
	return &EmailChannel{
		smtpHost: emailConfig.SMTPHost,
		smtpPort: emailConfig.SMTPPort,
		username: emailConfig.Username,
		password: emailConfig.Password,
	}
}

func (ec *EmailChannel) SendNotification(ctx context.Context, request *models.NotificationRequest) (*models.NotificationResponse, error) {
	log := logger.GetLogger()
	log.Info("Sending email notification", "to", request.Recipient, "subject", request.Subject)

	// Simulate email sending
	response := &models.NotificationResponse{
		MessageID:   fmt.Sprintf("email_%s", request.ID),
		Status:      "sent",
		Channel:     "email",
		Recipient:   request.Recipient,
		ChannelData: map[string]interface{}{
			"smtp_host": ec.smtpHost,
			"smtp_port": ec.smtpPort,
			"subject":   request.Subject,
		},
	}

	return response, nil
}

func (ec *EmailChannel) ValidateNotification(request *models.NotificationRequest) error {
	if request.Recipient == "" {
		return fmt.Errorf("recipient email is required")
	}
	if request.Subject == "" {
		return fmt.Errorf("email subject is required")
	}
	if request.Message == "" {
		return fmt.Errorf("email message is required")
	}
	return nil
}

func (ec *EmailChannel) GetChannelType() string {
	return "email"
}

func (ec *EmailChannel) GetChannelName() string {
	return "Email"
}

// SMSChannel implements NotificationChannel for SMS
type SMSChannel struct {
	apiKey    string
	apiSecret string
}

// NewSMSChannel creates a new SMS channel instance
func NewSMSChannel() *SMSChannel {
	cfg := config.GetConfigManager()
	smsConfig := cfg.GetSMSConfig()
	return &SMSChannel{
		apiKey:    smsConfig.APIKey,
		apiSecret: smsConfig.APISecret,
	}
}

func (sc *SMSChannel) SendNotification(ctx context.Context, request *models.NotificationRequest) (*models.NotificationResponse, error) {
	log := logger.GetLogger()
	log.Info("Sending SMS notification", "to", request.Recipient)

	// Simulate SMS sending
	response := &models.NotificationResponse{
		MessageID:   fmt.Sprintf("sms_%s", request.ID),
		Status:      "sent",
		Channel:     "sms",
		Recipient:   request.Recipient,
		ChannelData: map[string]interface{}{
			"sms_provider": "twilio",
			"message_length": len(request.Message),
		},
	}

	return response, nil
}

func (sc *SMSChannel) ValidateNotification(request *models.NotificationRequest) error {
	if request.Recipient == "" {
		return fmt.Errorf("recipient phone number is required")
	}
	if request.Message == "" {
		return fmt.Errorf("SMS message is required")
	}
	if len(request.Message) > 160 {
		return fmt.Errorf("SMS message too long (max 160 characters)")
	}
	return nil
}

func (sc *SMSChannel) GetChannelType() string {
	return "sms"
}

func (sc *SMSChannel) GetChannelName() string {
	return "SMS"
}

// PushChannel implements NotificationChannel for Push Notifications
type PushChannel struct {
	serverKey string
}

// NewPushChannel creates a new Push channel instance
func NewPushChannel() *PushChannel {
	cfg := config.GetConfigManager()
	pushConfig := cfg.GetPushConfig()
	return &PushChannel{
		serverKey: pushConfig.ServerKey,
	}
}

func (pc *PushChannel) SendNotification(ctx context.Context, request *models.NotificationRequest) (*models.NotificationResponse, error) {
	log := logger.GetLogger()
	log.Info("Sending push notification", "to", request.Recipient)

	// Simulate push notification sending
	response := &models.NotificationResponse{
		MessageID:   fmt.Sprintf("push_%s", request.ID),
		Status:      "sent",
		Channel:     "push",
		Recipient:   request.Recipient,
		ChannelData: map[string]interface{}{
			"push_provider": "firebase",
			"device_token":  request.Recipient,
		},
	}

	return response, nil
}

func (pc *PushChannel) ValidateNotification(request *models.NotificationRequest) error {
	if request.Recipient == "" {
		return fmt.Errorf("device token is required")
	}
	if request.Title == "" {
		return fmt.Errorf("push notification title is required")
	}
	if request.Message == "" {
		return fmt.Errorf("push notification message is required")
	}
	return nil
}

func (pc *PushChannel) GetChannelType() string {
	return "push"
}

func (pc *PushChannel) GetChannelName() string {
	return "Push Notification"
}

// WhatsAppChannel implements NotificationChannel for WhatsApp
type WhatsAppChannel struct {
	apiKey    string
	apiSecret string
}

// NewWhatsAppChannel creates a new WhatsApp channel instance
func NewWhatsAppChannel() *WhatsAppChannel {
	cfg := config.GetConfigManager()
	whatsappConfig := cfg.GetWhatsAppConfig()
	return &WhatsAppChannel{
		apiKey:    whatsappConfig.APIKey,
		apiSecret: whatsappConfig.APISecret,
	}
}

func (wc *WhatsAppChannel) SendNotification(ctx context.Context, request *models.NotificationRequest) (*models.NotificationResponse, error) {
	log := logger.GetLogger()
	log.Info("Sending WhatsApp notification", "to", request.Recipient)

	// Simulate WhatsApp sending
	response := &models.NotificationResponse{
		MessageID:   fmt.Sprintf("whatsapp_%s", request.ID),
		Status:      "sent",
		Channel:     "whatsapp",
		Recipient:   request.Recipient,
		ChannelData: map[string]interface{}{
			"whatsapp_provider": "twilio",
			"message_type":      "text",
		},
	}

	return response, nil
}

func (wc *WhatsAppChannel) ValidateNotification(request *models.NotificationRequest) error {
	if request.Recipient == "" {
		return fmt.Errorf("WhatsApp number is required")
	}
	if request.Message == "" {
		return fmt.Errorf("WhatsApp message is required")
	}
	return nil
}

func (wc *WhatsAppChannel) GetChannelType() string {
	return "whatsapp"
}

func (wc *WhatsAppChannel) GetChannelName() string {
	return "WhatsApp"
}

// SlackChannel implements NotificationChannel for Slack
type SlackChannel struct {
	webhookURL string
	channel    string
}

// NewSlackChannel creates a new Slack channel instance
func NewSlackChannel() *SlackChannel {
	cfg := config.GetConfigManager()
	slackConfig := cfg.GetSlackConfig()
	return &SlackChannel{
		webhookURL: slackConfig.WebhookURL,
		channel:    slackConfig.Channel,
	}
}

func (slc *SlackChannel) SendNotification(ctx context.Context, request *models.NotificationRequest) (*models.NotificationResponse, error) {
	log := logger.GetLogger()
	log.Info("Sending Slack notification", "to", request.Recipient)

	// Simulate Slack sending
	response := &models.NotificationResponse{
		MessageID:   fmt.Sprintf("slack_%s", request.ID),
		Status:      "sent",
		Channel:     "slack",
		Recipient:   request.Recipient,
		ChannelData: map[string]interface{}{
			"slack_channel": slc.channel,
			"webhook_url":   slc.webhookURL,
		},
	}

	return response, nil
}

func (slc *SlackChannel) ValidateNotification(request *models.NotificationRequest) error {
	if request.Recipient == "" {
		return fmt.Errorf("Slack channel is required")
	}
	if request.Message == "" {
		return fmt.Errorf("Slack message is required")
	}
	return nil
}

func (slc *SlackChannel) GetChannelType() string {
	return "slack"
}

func (slc *SlackChannel) GetChannelName() string {
	return "Slack"
}
